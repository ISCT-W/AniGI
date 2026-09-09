"""Export only the exact accepted artifact and track separate user acceptance."""

import json
import os
from pathlib import Path
import shutil
import tempfile

from .config import use_workspace
from .workspace import TaskError, artifact_path, file_hash, refresh_index, save_json, task_lock


def _image_approval(task, round_id, candidate):
    from .image.task_store import TaskStore
    store = TaskStore(artifact_path(task, "image"))
    approved = store.approved_candidate(round_id, candidate)
    source = artifact_path(store.path, approved["image"]["path"])
    return source, approved


def _video_approval(task, record):
    from .video import runtime
    runs = artifact_path(task, "video")
    with runtime.locked(record["video_run_id"], runs=runs) as (_, _, state):
        approved = runtime.validate_final(state)
    source = artifact_path(task, Path(approved["observation"]["media_path"]))
    return source, approved


def _current(task, record, version):
    if record["purpose"] == "image":
        source, approval = _image_approval(task, version["round_id"], version["candidate"])
    else:
        source, approval = _video_approval(task, record)
    if source.relative_to(task).as_posix() != version["source"] or approval != version["approval"]:
        raise TaskError("The source approval changed; this delivery is no longer current")
    if file_hash(source) != version["sha256"]:
        raise TaskError("The approved source content changed")
    return source


def reconcile(task, record):
    """Withdraw stale approvals while retaining artifacts and user feedback."""
    # Status refresh is also a public entry point, outside CLI configuration scope.
    # All approval readers must resolve settings against the owning workspace.
    with use_workspace(Path(task).parent.parent.parent):
        return _reconcile(task, record)


def _reconcile(task, record):
    path = artifact_path(task, "delivery.json")
    manifest = json.loads(path.read_text())
    for row in manifest["versions"]:
        try:
            _current(task, record, row)
            output = artifact_path(task, "final_output/" + row["name"])
            valid = file_hash(output) == row["sha256"]
        except (ValueError, OSError, KeyError):
            valid = False
        row["approval_valid"] = valid
    current = [row["name"] for row in manifest["versions"]
               if row["approval_valid"] and row["acceptance"] != "changes_requested"]
    manifest["current_version"] = current[-1] if current else None
    save_json(path, manifest)
    _write_acceptance(task, manifest)
    return manifest


def export(task, round_id=None, candidate=None):
    with task_lock(task) as (path, root, record), use_workspace(root):
        if record["purpose"] == "image":
            if not round_id or not candidate:
                raise TaskError("Image delivery requires a round and candidate")
            source, approval = _image_approval(path, round_id, candidate)
        else:
            if round_id or candidate:
                raise TaskError("Video delivery exports the accepted full film, not an image candidate")
            source, approval = _video_approval(path, record)
        suffix = source.suffix.lower()
        if record["purpose"] == "video" and suffix != ".mp4":
            raise TaskError("Expected the accepted full-film MP4")
        if record["purpose"] == "image" and suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise TaskError("Expected an accepted image")
        digest = file_hash(source)
        manifest_path = artifact_path(path, "delivery.json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": 1, "versions": []}
        for row in manifest["versions"]:
            if row["source"] == source.relative_to(path).as_posix() and row["approval"] == approval:
                output = artifact_path(path, "final_output/" + row["name"])
                if file_hash(output) != row["sha256"]:
                    raise TaskError("An existing delivery changed")
                refresh_index(path)
                return output
        row = {"source": source.relative_to(path).as_posix(), "sha256": digest,
               "approval": approval, "acceptance": "pending", "feedback": [],
               "round_id": round_id, "candidate": candidate, "approval_valid": True}
        fd, staged_name = tempfile.mkstemp(prefix=".delivery-", dir=path)
        staged = Path(staged_name)
        try:
            with source.open("rb") as incoming, os.fdopen(fd, "wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if file_hash(staged) != digest:
                raise TaskError("Source changed during delivery; no final artifact was committed")
            _current(path, record, row)
            directory = artifact_path(path, "final_output")
            directory.mkdir(exist_ok=True)
            number = len(manifest["versions"]) + 1
            while True:
                name = f"final-v{number:03d}{suffix}"
                output = artifact_path(path, "final_output/" + name)
                if output.exists():
                    # A crash after linking but before the manifest commit is recoverable.
                    if not any(item["name"] == name for item in manifest["versions"]) and file_hash(output) == digest:
                        break
                    number += 1
                    continue
                os.link(staged, output)
                break
        finally:
            staged.unlink(missing_ok=True)
        row["name"] = name
        manifest["versions"].append(row)
        manifest["current_version"] = name
        save_json(manifest_path, manifest)
        _write_acceptance(path, manifest)
        refresh_index(path)
        return output


def accept(task, version, status, comment):
    if status not in {"accepted", "changes_requested"} or not isinstance(comment, str) or not comment.strip():
        raise TaskError("Record an explicit user verdict and original comment")
    with task_lock(task) as (path, root, record), use_workspace(root):
        manifest_path = artifact_path(path, "delivery.json")
        if not manifest_path.is_file():
            raise TaskError("No exported versions are available for user acceptance")
        manifest = json.loads(manifest_path.read_text())
        row = next((item for item in manifest["versions"] if item["name"] == version), None)
        if row is None:
            raise TaskError("Unknown delivery version")
        output = artifact_path(path, "final_output/" + row["name"])
        if file_hash(output) != row["sha256"]:
            raise TaskError("The delivered file changed")
        if status == "accepted":
            _current(path, record, row)
        row["acceptance"] = status
        row["feedback"].append({"status": status, "comment": comment})
        save_json(manifest_path, manifest)
        _write_acceptance(path, manifest)
        refresh_index(path)
        return row


def _write_acceptance(task, manifest):
    rows = ["# Acceptance", "", "Agent approval and user acceptance are separate.", ""]
    for version in manifest["versions"]:
        validity = "current" if version.get("approval_valid") else "historical; approval withdrawn or needs revalidation"
        rows += [f"## {version['name']}", "", f"Agent approval: {validity}", f"User: {version['acceptance']}", ""]
        for feedback in version["feedback"]:
            rows += [f"Verdict: {feedback['status']}", "", *("> " + line for line in feedback["comment"].splitlines()), ""]
    artifact_path(task, "final_output/acceptance.md").write_text("\n".join(rows), encoding="utf-8")
