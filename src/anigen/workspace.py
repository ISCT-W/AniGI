"""Task naming, containment and a readable index for private production records."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import quote
import uuid

from .config import ConfigError, read_settings, use_workspace


TOKYO = timezone(timedelta(hours=9))
BACKENDS = {"gpt", "gemini"}
PURPOSES = {"image": "figs", "video": "videos"}


class TaskError(ValueError):
    pass


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def description_slug(value):
    if not isinstance(value, str):
        raise TaskError("Task description must be text")
    value = unicodedata.normalize("NFC", value).strip()
    value = "".join("-" if unicodedata.category(char).startswith("C") else char for char in value)
    value = re.sub(r'[\\/:*?"<>|\s]+', "-", value)
    value = value.strip(" .-")[:48].rstrip(" .-")
    return value or "task"


def _contained(root, path):
    root, path = Path(root).resolve(), Path(path)
    try:
        relative = path.absolute().relative_to(root)
    except ValueError:
        raise TaskError("Task artifact is outside its workspace") from None
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise TaskError("Task paths cannot traverse symbolic links")
    if not path.resolve().is_relative_to(root):
        raise TaskError("Task artifact resolves outside its workspace")
    return path.resolve()


def create_task(workspace, purpose, description, backend=None, brief="", mode="offline", authorization="", now=None):
    root = Path(workspace).expanduser().resolve()
    if purpose not in PURPOSES or mode not in {"offline", "generation"}:
        raise TaskError("Choose image/video purpose and offline/generation mode")
    if not isinstance(brief, str) or not brief.strip():
        raise TaskError("A task brief is required")
    if mode == "generation" and (not isinstance(authorization, str) or not authorization.strip()):
        raise TaskError("Record the user's explicit generation instruction")
    with use_workspace(root):
        if backend is None:
            backend = read_settings(["IMAGE_BACKEND"]).get("IMAGE_BACKEND")
        if backend not in BACKENDS:
            raise TaskError("Select gpt or gemini explicitly, or configure IMAGE_BACKEND")
        scope = "offline-fixture"
        if mode == "generation":
            scope = read_settings(["REFERENCE_SCOPE_ID"]).get("REFERENCE_SCOPE_ID", "")
            if not scope:
                raise ConfigError("Configure REFERENCE_SCOPE_ID before a production task")
        base = _contained(root, root / "generation" / PURPOSES[purpose])
        base.mkdir(parents=True, exist_ok=True)
        stamp = now or datetime.now(TOKYO)
        if stamp.tzinfo is None:
            raise TaskError("Task creation time must include a timezone")
        stamp = stamp.astimezone(TOKYO)
        description = description_slug(description)
        label = backend + ("-minimax" if purpose == "video" else "")
        for _ in range(10000):
            time_field = stamp.strftime("%Y%m%dT%H%M%S%f%z")
            path = base / f"{time_field}*{description}*{label}"
            try:
                path.mkdir(mode=0o700)
                break
            except FileExistsError:
                stamp = max(datetime.now(TOKYO), stamp + timedelta(microseconds=1))
        else:
            raise TaskError("Unable to allocate a unique task directory")
        (path / "brief.md").write_text(brief, encoding="utf-8")
        authorization = authorization or "Offline task; no live calls authorized.\n"
        (path / "authorization.md").write_text(authorization, encoding="utf-8")
        (path / "feedback.md").write_text("# User feedback\n\nNo user acceptance recorded.\n", encoding="utf-8")
        (path / "references").mkdir()
        if purpose == "video":
            (path / "director").mkdir()
        task_id = uuid.uuid4().hex
        record = {
            "schema": 1, "id": task_id, "purpose": purpose, "description": description,
            "created_at": stamp.isoformat(), "time_field": time_field, "initial_backend": backend,
            "tool_label": label, "mode": mode, "reference_scope_id": scope,
            "video_run_id": "v_" + task_id[:16] if purpose == "video" else None,
            "brief": {"path": "brief.md", "sha256": file_hash(path / "brief.md")},
            "authorization": {"path": "authorization.md", "sha256": file_hash(path / "authorization.md")},
        }
        if purpose == "image":
            from .image.task_store import TaskStore
            store = TaskStore.create(path / "image", brief, description, mode=mode, authorization=authorization,
                                     limit=6, project_id=scope, initial_backend=backend)
            record["image_stage_id"] = store.snapshot()["id"]
        save_json(path / "task.json", record)
        refresh_index(path)
        return path


def load_task(path):
    raw = Path(path).expanduser().absolute()
    if raw.parent.name not in PURPOSES.values() or raw.parent.parent.name != "generation":
        raise TaskError("Task must be inside generation/figs or generation/videos")
    root = raw.parent.parent.parent.resolve()
    path = _contained(root, raw)
    try:
        record = json.loads(_contained(path, path / "task.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise TaskError("Task metadata is missing or invalid") from None
    purpose = record.get("purpose")
    if record.get("schema") != 1 or purpose not in PURPOSES or path.parent.name != PURPOSES[purpose]:
        raise TaskError("Task purpose or schema does not match its directory")
    if record.get("initial_backend") not in BACKENDS:
        raise TaskError("Task initial backend is invalid")
    expected_label = record["initial_backend"] + ("-minimax" if purpose == "video" else "")
    expected_name = f"{record.get('time_field')}*{record.get('description')}*{expected_label}"
    if record.get("tool_label") != expected_label or path.name != expected_name:
        raise TaskError("Task directory no longer matches its frozen naming record")
    if not re.fullmatch(r"[a-f0-9]{32}", record.get("id", "")):
        raise TaskError("Task ID is invalid")
    if record.get("mode") not in {"offline", "generation"}:
        raise TaskError("Task mode is invalid")
    for name in ("brief", "authorization"):
        item = record.get(name, {})
        if item.get("path") != name + ".md":
            raise TaskError("Task document locator changed")
        target = _contained(path, path / item["path"])
        if not target.is_file() or file_hash(target) != item.get("sha256"):
            raise TaskError("Frozen task document changed")
    if purpose == "image":
        from .image.task_store import TaskStore
        with use_workspace(root):
            state = TaskStore(_contained(path, path / "image")).snapshot()
        if (state.get("id") != record.get("image_stage_id")
                or state.get("mode") != record["mode"]
                or state.get("project_id") != record["reference_scope_id"]
                or state.get("initial_backend") != record["initial_backend"]
                or state.get("brief", {}).get("sha256") != record["brief"]["sha256"]
                or state.get("batches", [{}])[0].get("authorization", {}).get("sha256") != record["authorization"]["sha256"]):
            raise TaskError("Image stage no longer belongs to this task's frozen identity and instructions")
    else:
        if record.get("video_run_id") != "v_" + record["id"][:16]:
            raise TaskError("Video run identity changed")
        state_path = _contained(path, path / "video" / record["video_run_id"] / "state.json")
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (state.get("id") != record["video_run_id"]
                    or state.get("plan", {}).get("task_id") != record["id"]
                    or state.get("plan", {}).get("reference_scope_id") != record["reference_scope_id"]):
                raise TaskError("Video run no longer belongs to this task's identity and scope")
    return path, root, record


@contextmanager
def task_lock(path):
    path, root, record = load_task(path)
    with artifact_path(path, ".task.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield path, root, record
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def artifact_path(task, relative):
    return _contained(Path(task).resolve(), Path(task) / relative)


def _record_block(value):
    # Indentation renders arbitrary prompts as text without interpreting Markdown.
    return ["    " + line for line in json.dumps(value, ensure_ascii=False, indent=2).splitlines()] + [""]


def _video_details(task, state):
    lines = []
    for attempt in state.get("attempts", []):
        lines += [f"### {attempt['id']}", "", f"Generation model: {attempt.get('endpoint', 'unknown')}",
                  "Generation duration and unreported costs: unknown.", "", "Complete request:", ""]
        lines += _record_block(attempt.get("arguments", {}))
        lines += ["Director timeline, correction and revision reasons:", ""]
        lines += _record_block({key: attempt.get(key) for key in ("director_timeline", "correction", "revision")})
        lines += ["Review history:", ""] + _record_block(attempt.get("reviews", []))
        for frame in attempt.get("observation", {}).get("frames", []):
            frame_path = artifact_path(task, Path(frame["path"]))
            link = quote(frame_path.relative_to(task).as_posix(), safe="/")
            lines += [f"![Observed frame]({link})", ""]
        result = attempt.get("result", {}) or {}
        lines += ["Provider-reported usage and cost:", ""]
        lines += _record_block({key: result.get(key) if result.get(key) is not None else "unknown"
                                for key in ("usage", "cost")})
    observations = state.get("av_observations", {})
    if observations:
        lines += ["### Complete visual observations", ""]
        for identifier, observation in observations.items():
            result = observation.get("result", {}) or {}
            requested = observation.get("request", {}).get("requested_model")
            lines += [f"#### {identifier}", ""]
            lines += _record_block({"target": observation.get("request", {}).get("target_id"),
                "status": observation.get("status"), "requested_model": requested,
                "returned_model": result.get("returned_model", "unknown"),
                "started_at": observation.get("started_at"), "completed_at": observation.get("completed_at"),
                "usage": observation.get("usage") or "unknown",
                "estimated_cost": observation.get("estimated_cost") if observation.get("estimated_cost") is not None else "unknown"})
    if state.get("final"):
        lines += ["### Full-film review history", ""] + _record_block(state["final"].get("reviews", []))
    return lines


def refresh_index(task):
    """Render summaries and local links; never copy raw service responses."""
    path, _, record = load_task(task)
    lines = [f"# {record['description']}", "", f"Final purpose: {record['purpose']}",
             f"Created: {record['created_at']}", f"Initial tool chain: {record['tool_label']}",
             f"Mode: {record['mode']}", "", "[Original brief](brief.md) · [User feedback](feedback.md)", ""]
    image_dir = "image" if record["purpose"] == "image" else "keyframe"
    image_state = artifact_path(path, image_dir + "/state.json")
    if image_state.is_file():
        state = json.loads(image_state.read_text())
        charged = sum(item["status"] not in {"prepared", "not_sent"} for item in state.get("attempts", []))
        lines += ["## Image stage", "", f"Requests reserved or sent: {charged} / 6",
                  f"[All image rounds, inputs and reviews]({image_dir}/README.md)", ""]
        for attempt in state.get("attempts", []):
            lines.append(f"- Round {attempt['id']}: {attempt['status']} · {attempt.get('backend', 'unknown')} / {attempt.get('model', 'unknown')}")
        lines.append("")
    if record["purpose"] == "video":
        relative = f"video/{record['video_run_id']}"
        video_state = artifact_path(path, relative + "/state.json")
        if video_state.is_file():
            state = json.loads(video_state.read_text())
            lines += ["## Video stage", "", "| Attempt | Shot | Status | Media and review records |", "| --- | --- | --- | --- |"]
            for item in state.get("attempts", []):
                media = item.get("observation", {}).get("media_path")
                link = "pending"
                if media:
                    media_path = artifact_path(path, Path(media))
                    relative_media = quote(media_path.relative_to(path).as_posix(), safe="/")
                    link = f"[Video]({relative_media})"
                lines.append(f"| {item['id']} | {item.get('shot_id', '')} | {item['status']} | {link} |")
            lines += ["", f"[Execution and review state]({relative}/state.json)", ""]
            lines += _video_details(path, state)
    manifest = artifact_path(path, "delivery.json")
    if manifest.is_file():
        from .delivery import reconcile
        rows = reconcile(path, record).get("versions", [])
        lines += ["## Deliveries", ""]
        for item in rows:
            status = "current approval" if item.get("approval_valid") else "historical; approval withdrawn or needs revalidation"
            if item.get("acceptance") == "changes_requested":
                status = "historical; user requested changes"
            lines.append(f"- [{item['name']}](final_output/{quote(item['name'])}): {status}; user {item['acceptance']}")
        lines.append("")
    else:
        lines += ["No final deliverable exported. Stage acceptance is not user acceptance.", ""]
    lines += ["Usage and costs: see each request receipt; unreported costs are unknown.", ""]
    artifact_path(path, "README.md").write_text("\n".join(lines), encoding="utf-8")
    return path / "README.md"
