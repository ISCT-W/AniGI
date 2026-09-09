"""Deterministic execution only. Codex supplies plans, evidence and review judgments.

Local JSON is a durable audit record, not an authentication/security boundary.
Single observation calls are tools, with no hidden planning loop or paid retries.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from .settings import get_setting
from pathlib import Path
import re
import shutil
import sys
import tempfile

from . import directing, handoff, keyframes, media, normalization, observation as av
from .provider import FalProvider, ProviderError, validate_request, request_duration, _https_url

RUNS = None  # Every caller supplies the task-local video directory.
CHECK_MODES = {
    **dict.fromkeys(("identity", "appearance", "composition", "scene", "prompt_adherence",
                     "continuity", "intent", "visual_artifacts"), "sampled_frames"),
    **dict.fromkeys(("temporal", "motion", "action", "action_timing", "camera", "pacing", "narrative"), "video"),
    **dict.fromkeys(("audio", "dialogue", "av_sync", "lip_sync"), "video_and_audio"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def ident(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value), "invalid ID")
    return value


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, state):
    """Replace one complete snapshot, including its event log, atomically."""
    handle, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def event(state, kind, **data):
    state["events"].append({"at": now(), "kind": kind, **data})


@contextmanager
def locked(run_id, runs=RUNS, create=False):
    require(runs is not None, "explicit task-local video directory required")
    directory = Path(runs).resolve() / ident(run_id)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    require(directory.is_dir() and not directory.is_symlink(), "run does not exist or is a symlink")
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "state.json"
        state = json.loads(path.read_text()) if path.exists() else None
        yield directory, path, state


def validate_plan(data):
    require(nonempty(data.get("brief")) and nonempty(data.get("script")), "brief and script required")
    require(nonempty(data.get("reference_scope_id")), "select a verified reference scope ID")
    require(data.get("sequence_mode", "independent") in ("independent", "reviewed_segments"), "invalid sequence_mode")
    shots = data.get("shots")
    require(isinstance(shots, list) and shots, "nonempty shots required")
    seen = set()
    for shot in shots:
        sid = ident(shot["id"])
        require(sid not in seen, "duplicate shot ID")
        seen.add(sid)
        require(nonempty(shot.get("goal")) and type(shot.get("duration_s")) is int and 5 <= shot["duration_s"] <= 15, "shot goal and integer duration 5..15 required")
        checks = shot.get("required_checks")
        require(isinstance(checks, list) and checks and all(nonempty(x) for x in checks), "required_checks must name acceptance criteria")
        require(all(c in CHECK_MODES for c in checks), "use canonical check IDs; put Chinese labels/details in reasons")
        require(isinstance(shot.get("evidence_ids"), list) and shot["evidence_ids"], "each shot needs reference evidence IDs")
    require(isinstance(data.get("final_checks"), list) and data["final_checks"] and all(nonempty(x) for x in data["final_checks"]), "final_checks required")
    require(all(c in CHECK_MODES for c in data["final_checks"]), "use canonical final check IDs")
    directing.validate(data)


def shot_for(state, shot_id):
    return next(s for s in state["plan"]["shots"] if s["id"] == shot_id)


def current_shot(state):
    return next((s for s in state["plan"]["shots"] if s["id"] not in state["accepted"]), None)


def last_attempt(state, shot_id):
    return next((a for a in reversed(state["attempts"]) if a["shot_id"] == shot_id), None)


def get_attempt(state, attempt_id):
    return next(a for a in state["attempts"] if a["id"] == attempt_id)


def check_review(report, observation, required_checks, *, artifacts=None, human_records=None, context=None, policy=None):
    require(all(c in CHECK_MODES for c in required_checks), "unknown required check ID")
    require(report.get("media_sha256") == observation["media_sha256"], "review belongs to different media")
    frames = {f["path"] for f in observation["frames"]}
    require(set(report.get("observed_frames", [])) == frames and frames, "review must attest to viewing every sampled frame")
    require(nonempty(report.get("summary")), "review summary required")
    mode = report.get("observation_mode")
    require(mode in ("sampled_frames", "video", "video_and_audio"), "invalid observation mode")
    checks = report.get("checks", {})
    for name, check in checks.items():
        require(name in CHECK_MODES, "unknown check ID: use documented canonical names")
        require(check.get("status") in ("pass", "fail", "unknown", "na") and nonempty(check.get("reason")), "checks need status and evidence-based reason")
        if CHECK_MODES[name] == "video" and check["status"] == "pass":
            require(mode != "sampled_frames", "sampled frames cannot clear temporal review")
        if CHECK_MODES[name] == "video_and_audio" and check["status"] == "pass":
            require(mode == "video_and_audio", "audio/lip-sync review requires audiovisual observation")
    if report.get("review_authority") == "codex_director" or mode != "sampled_frames" or any(c in av.DYNAMIC_CHECKS and v.get("status") == "pass" for c, v in checks.items()):
        require(context is not None, "dynamic review needs persisted observation context; playback_evidence is insufficient")
        av.verify_dynamic_passes(report, artifacts or {}, human_records or {}, context, policy)
    decision = report.get("decision")
    require(decision in ("accept", "reject", "needs_review"), "invalid review decision")
    if decision == "accept":
        require(all(checks.get(c, {}).get("status") == "pass" for c in required_checks), "required checks have not passed")
        require(not any(c["status"] == "fail" for c in checks.values()), "cannot accept a failed check")
        require(nonempty(report.get("continuity_state")), "accepted result needs continuity_state")
    if decision == "reject":
        require(nonempty(report.get("correction")), "rejection needs a specific correction")


def observe(path, directory, shot=None, boundary_times=None):
    qc = media.inspect_video(path, expected_duration=shot["duration_s"] if shot else None,
                             expected_aspect=shot.get("aspect_ratio") if shot else None,
                             require_audio=shot.get("require_audio", False) if shot else False)
    result = {"qc": qc, "media_path": str(path), "media_sha256": digest(path), "frames": []}
    if qc["ok"]:
        sample = media.extract_frames(path, directory, boundary_times=boundary_times)
        result["frames"] = [{**frame, "sha256": digest(frame["path"])} for frame in sample["frames"]]
        result["limitations"] = sample.get("limitations", [])
        result["signal_stats"] = media.signal_stats(path)
    return result


def verify_observation(observation):
    require(digest(observation["media_path"]) == observation["media_sha256"], "media changed since observation")
    for frame in observation["frames"]:
        require(digest(frame["path"]) == frame["sha256"], "sample frame changed since observation")


def verify_normalization(attempt):
    """A technical derivative keeps its immutable source and generation provenance."""
    for record in attempt.get("normalizations", []):
        require(record["attempt_id"] == attempt["id"] and record["shot_id"] == attempt["shot_id"],
                "normalization belongs to another attempt")
        require(digest(record["manifest_path"]) == record["manifest_sha256"], "normalization manifest changed")
        package = {key: value for key, value in record.items() if key not in ("manifest_path", "manifest_sha256")}
        require(av.content_hash(json.loads(Path(record["manifest_path"]).read_text())) == av.content_hash(package),
                "normalization state differs from its immutable manifest")
        verify_observation(record["original_observation"])
        require(av.content_hash(record["derived_observation"]) == av.content_hash(attempt["observation"]),
                "normalization no longer matches the current derived observation")
        require(all(attempt.get(key) == value for key, value in record["generation_provenance"].items()),
                "normalization generation request, references or handoff changed")


OBSERVATION_ACTIONS = {"observation-policy", "import-video", "observe-av", "inspect-window",
                       "cleanup-observation", "record-human-review", "reopen-review"}


def review_context(state, target_id):
    if target_id == "final":
        target = state["final"]
        require(target is not None, "final is not assembled")
        specs = {"plan": state["plan"], "accepted_attempts": state["accepted"]}
        ids = sorted({eid for shot in state["plan"]["shots"] for eid in shot["evidence_ids"]})
    else:
        target = get_attempt(state, target_id)
        shot = shot_for(state, target["shot_id"])
        specs = {"plan": state["plan"], "shot_id": shot["id"],
                 "generation_arguments": target.get("arguments"), "correction": target.get("correction")}
        ids = shot["evidence_ids"]
        if state["plan"].get("sequence_mode") == "reviewed_segments":
            specs.update(handoff_id=target.get("handoff_id"), reference_bindings=target.get("reference_bindings", {}))
            ids = sorted(set(ids) | set(target.get("reference_bindings", {}).values()))
    require("observation" in target, "target has no collected video")
    observation = target["observation"]
    verify_observation(observation)
    require(observation["qc"]["ok"], "target failed technical QC")
    refs = {eid: state["evidence"][eid] for eid in ids}
    for value in refs.values():
        if value.get("observation_path"):
            require(digest(value["observation_path"]) == value.get("observation_sha256"), "reference file changed")
    context = {"source_hash": observation["media_sha256"], "spec_hash": av.content_hash(specs),
               "reference_hash": av.content_hash(refs), "target_id": target_id, "run_id": state["id"]}
    return target, context, refs


def av_review(state, report, target_id, checks):
    if target_id != "final":
        verify_normalization(get_attempt(state, target_id))
    target, context, refs = review_context(state, target_id)
    directing.verify(state)
    directing.review(state, report)
    if report.get("review_authority") == "codex_director":
        require(set(report.get("director_review", {}).get("reference_ids", [])) == set(refs),
                "director review must attest all current original and continuity references")
    check_review(report, target["observation"], checks, artifacts=state.get("av_observations", {}),
                 human_records=state.get("human_reviews", {}), context=context, policy=state.get("observation_policy"))
    if target_id == "final" and report.get("decision") == "accept" and any(c in av.DYNAMIC_CHECKS for c in checks):
        for boundary in target.get("boundaries_s", []):
            reviews = report.get("boundary_reviews", [])
            item = next((r for r in reviews if r.get("boundary_s") == boundary), None)
            require(item and nonempty(item.get("reason")), "final dynamic review needs each seam inspected")
            window = state.get("inspection_windows", {}).get(item.get("window_id"))
            require(window and all(window[k] == context[k] for k in context), "seam window belongs to old media")
            clip = window["media"]
            require(clip["requested_window_s"]["start"] < boundary < clip["requested_window_s"]["end"], "seam outside window")
            require(digest(clip["crop"]["path"]) == clip["crop"]["sha256"], "seam crop changed")
            frames = clip["frames"]
            require(set(item.get("observed_frames", [])) == {f["path"] for f in frames}, "view every seam window frame")
            require(any(f["source_timestamp_s"] < boundary for f in frames) and any(f["source_timestamp_s"] >= boundary for f in frames),
                    "seam frames must cover both sides")
            require(all(digest(f["path"]) == f["sha256"] for f in frames), "seam frame changed")


def verify_checkpoints(state):
    """Historical accepted flags do not bypass current evidence requirements."""
    for sid, aid in state["accepted"].items():
        target = get_attempt(state, aid)
        keyframes.gate(state, target)
        require(target["status"] == "accepted" and target["shot_id"] == sid, "invalid accepted checkpoint")
        report = next((r for r in reversed(target.get("reviews", [])) if r.get("decision") == "accept"), None)
        require(report is not None, "accepted checkpoint has no review; reopen-review required")
        av_review(state, report, aid, shot_for(state, sid)["required_checks"])
        verify_segment_handoff(state, target)


def validate_final(state):
    """Revalidate the exact current final media, all segments and seam reviews."""
    verify_checkpoints(state)
    final = state.get('final')
    require(final and final.get('status') == 'accepted', 'final video is not accepted')
    report = final.get('reviews', [])[-1] if final.get('reviews') else None
    require(report and report.get('decision') == 'accept', 'final video lacks current acceptance review')
    verify_observation(final['observation'])
    av_review(state, report, 'final', state['plan']['final_checks'])
    return json.loads(json.dumps({**final, 'review_sha256': av.content_hash(report)}))


def verify_handoff(state, handoff_id):
    record = state.get("handoffs", {}).get(handoff_id)
    require(record is not None, "unknown handoff ID; export an accepted segment with handoff first")
    require(state["accepted"].get(record["shot_id"]) == record["target_id"], "handoff source is no longer the accepted segment")
    target, context, _ = review_context(state, record["target_id"])
    require(target["status"] == "accepted", "handoff source is not accepted")
    require(all(record.get(key) == value for key, value in context.items()), "handoff belongs to old media/specification/references")
    report = target.get("reviews", [])[-1]
    require(report.get("decision") == "accept" and av.content_hash(report) == record["accepted_review_sha256"],
            "handoff acceptance review changed; export a fresh handoff")
    av_review(state, report, target["id"], shot_for(state, target["shot_id"])["required_checks"])
    manifest = Path(record["manifest_path"])
    require(digest(manifest) == record["manifest_sha256"], "handoff manifest changed")
    package = {key: value for key, value in record.items() if key not in ("manifest_path", "manifest_sha256")}
    require(av.content_hash(json.loads(manifest.read_text())) == av.content_hash(package), "handoff record differs from immutable manifest")
    for artifact in [*record["artifacts"].values(), *record["ending"]["frames"]]:
        require(digest(artifact["path"]) == artifact["sha256"], "handoff artifact changed")
    return record


def verify_segment_handoff(state, attempt):
    """The payload must carry the immediate accepted predecessor's real ending."""
    if state["plan"].get("sequence_mode") != "reviewed_segments":
        return
    ids = [shot["id"] for shot in state["plan"]["shots"]]
    index = ids.index(attempt["shot_id"])
    if index == 0:
        require(not attempt.get("handoff_id"), "the first segment has no predecessor handoff")
        return
    predecessor = state["accepted"].get(ids[index - 1])
    require(predecessor is not None, "previous segment must be accepted before continuing")
    require(nonempty(attempt.get("handoff_id")), "next segment requires handoff_id and an uploaded final-frame or tail-video reference binding")
    record = verify_handoff(state, attempt["handoff_id"])
    require(record["target_id"] == predecessor and record["shot_id"] == ids[index - 1], "handoff is not from the immediately preceding accepted segment")
    args, endpoint = attempt.get("arguments", {}), attempt.get("endpoint", "")
    if endpoint.endswith("image-to-video"):
        candidates = [(args.get("image_url"), "final_frame")]
    elif endpoint.endswith("reference-to-video"):
        candidates = [(url, "final_frame") for url in args.get("reference_image_urls", [])]
        candidates += [(url, "tail_video") for url in args.get("reference_video_urls", [])]
    else:
        candidates = []
    bindings = attempt.get("reference_bindings", {})
    matched = False
    for url, kind in candidates:
        evidence = state["evidence"].get(bindings.get(url), {})
        if evidence.get("handoff_id") != record["id"] or evidence.get("handoff_artifact") != kind:
            continue
        artifact = record["artifacts"][kind]
        require(evidence.get("verification") == "verified" and evidence.get("generation_url") == url,
                "handoff URL is not bound to verified uploaded evidence")
        require(evidence.get("observation_path") == artifact["path"] and evidence.get("observation_sha256") == artifact["sha256"]
                and evidence.get("observation_mime_type") == artifact["mime_type"], "handoff evidence is not bound to the exported artifact")
        require(digest(evidence["observation_path"]) == artifact["sha256"], "handoff reference file changed")
        if kind == "tail_video":
            require(2 <= artifact["duration_s"] <= 15 and evidence.get("duration_s") == artifact["duration_s"],
                    "fal tail-video reference needs at least 2 seconds; export tail_duration_s=2 or use the final frame")
            require(artifact["contains_final_frame"] and abs(artifact["source_end_s"] - record["ending"]["source_duration_s"]) < 0.000002,
                    "handoff tail video must reach the accepted source ending")
        matched = True
    require(matched, "payload must use the uploaded handoff final_frame as image_url/reference_image_urls or tail_video as reference_video_urls, with exact evidence binding")


def handle_observation(action, state, directory, path, data, live, observer):
    """Called under the run lock. Persist before every potentially billable call."""
    from .gemini_observer import GeminiObserver, PROMPT_VERSION, DEFAULT_MODEL, SYSTEM_PROMPT_SHA256
    records = state.setdefault("av_observations", {})
    if action == "observation-policy":
        data.setdefault("reserve_final_media_seconds", sum(s["duration_s"] for s in state["plan"]["shots"]) * data.get("reserve_final_calls", 1))
        policy = av.validate_policy(data)
        require(nonempty(state["plan"]["reference_scope_id"]), "observation requires a selected reference scope")
        require(set(policy["reference_transfer_ids"]) <= set(state["evidence"]), "unknown authorized reference IDs")
        policy.update(active=True, at=now())
        state["observation_policy"] = policy
        event(state, "observation_policy_recorded", policy=policy)
        return
    if action == "import-video":
        verify_checkpoints(state)
        shot = current_shot(state)
        require(shot and shot["id"] == data.get("shot_id"), "import only the next unaccepted shot")
        require(state["plan"].get("sequence_mode") != "reviewed_segments" or shot["id"] == state["plan"]["shots"][0]["id"],
                "reviewed_segments can import its first segment only; downstream generation must record actual handoff payload binding")
        require(nonempty(data.get("user_instruction")), "record user instruction for importing the existing clip")
        previous = last_attempt(state, shot["id"])
        require(previous is None or previous["status"] in ("rejected", "failed", "superseded"), "resolve current attempt first")
        require(all(state["evidence"].get(eid, {}).get("verification") == "verified" for eid in shot["evidence_ids"]),
                "imported shot reference evidence must be verified")
        source = Path(data["media_path"]).expanduser().resolve()
        require(source.is_file() and 0 < source.stat().st_size <= 100 * 1024 * 1024, "import file must be 1 byte..100 MiB")
        aid = f"attempt_{len(state['attempts']) + 1:04d}"
        output = Path(tempfile.mkdtemp(prefix=aid + "-import-", dir=directory))
        shutil.copyfile(source, output / "video.mp4")
        observed = observe(output / "video.mp4", output / "frames", shot)
        require(observed["qc"]["ok"], "imported video failed shot QC")
        state["attempts"].append({"id": aid, "shot_id": shot["id"], "status": "review_pending",
                                  "charged": False, "imported": True, "user_instruction": data["user_instruction"],
                                  "observation": observed})
        event(state, "video_imported", attempt_id=aid, source_sha256=observed["media_sha256"])
        return
    if action == "cleanup-observation":
        require(live, "remote file cleanup requires --live")
        artifact = records[ident(data["observation_id"])]
        if artifact["status"] == "in_progress":
            require(nonempty(data.get("recovery_note")), "record why interrupted observation is being reconciled")
        client = observer or GeminiObserver()
        client.preflight()
        manifest = directory / artifact["id"] / "gemini-remote-files.json"
        persisted = json.loads(manifest.read_text()) if manifest.exists() else {}
        require(isinstance(persisted, dict), "invalid persisted remote-file manifest")
        owned = list({record["name"]: record for record in [*artifact.get("remote_files", []), *persisted.get("remote_files", [])]}.values())
        artifact["remote_upload_outcome_unknown"] = bool(artifact.get("remote_upload_outcome_unknown") or persisted.get("remote_upload_outcome_unknown"))
        result = client.cleanup(owned)
        artifact["remote_files"] = result["remote_files"]
        artifact["cleanup_status"] = result["cleanup_status"]
        if artifact.get("remote_upload_outcome_unknown"):
            artifact["cleanup_status"] = "cleanup_pending"
        # A crashed call stays charged. This explicit action resolves it for a
        # later bounded retry, without claiming the earlier call was free.
        if artifact["status"] == "in_progress":
            require(nonempty(data.get("recovery_note")), "record why interrupted observation is being reconciled")
            artifact.update(status="interrupted", recovery_note=data["recovery_note"])
        event(state, "observation_cleanup", observation_id=artifact["id"], status=artifact["cleanup_status"])
        return
    target_id = ident(data.get("target_id", data.get("attempt_id", "final")))
    target, context, refs = review_context(state, target_id)
    if action == "reopen-review":
        require(target_id != "final" and target["status"] == "accepted", "reopen an accepted shot")
        require(nonempty(data.get("reason")), "reopening reason required")
        require(not any(a["status"] in ("submission_unknown", "submitted", "completed", "review_pending") for a in state["attempts"]),
                "resolve active attempt before reopening a checkpoint")
        ids = [s["id"] for s in state["plan"]["shots"]]
        affected = ids[ids.index(target["shot_id"]):]
        for sid in affected:
            state["accepted"].pop(sid, None)
            for attempt in state["attempts"]:
                if attempt["shot_id"] == sid and attempt["status"] in ("accepted", "prepared"):
                    attempt["status"] = "superseded"
        target["status"] = "review_pending"
        if state["final"]:
            state.setdefault("final_history", []).append(state["final"])
        state["final"] = None
        event(state, "checkpoint_reopened", target_id=target_id, reason=data["reason"])
        return
    require(target["status"] == "review_pending", "target is not awaiting review")
    observation = target["observation"]
    if action == "record-human-review":
        require(nonempty(data.get("user_confirmation")) and nonempty(data.get("observation_description")),
                "human review requires real user confirmation and actual playback scope")
        checks = data.get("checks")
        require(isinstance(checks, list) and checks and set(checks) <= av.DYNAMIC_CHECKS,
                "human confirmation must name supported dynamic checks")
        require(not set(checks) & (av.AUDIO_CHECKS | {"lip_sync"}) or observation["qc"]["has_audio"], "audio stream absent")
        require(data.get("media_sha256") == context["source_hash"], "human confirmation media mismatch")
        hid = f"human_{len(state.setdefault('human_reviews', {})) + 1:04d}"
        state["human_reviews"][hid] = {**context, **{k: data[k] for k in ("checks", "user_confirmation", "observation_description")}, "at": now()}
        event(state, "human_review_recorded", human_review_id=hid)
        return
    if action == "inspect-window":
        start, end = data["start_s"], data["end_s"]
        output = Path(tempfile.mkdtemp(prefix="window-", dir=directory))
        window = media.inspect_window(Path(observation["media_path"]), output, start, end, data.get("fps", 8))
        wid = f"window_{len(state.setdefault('inspection_windows', {})) + 1:04d}"
        state["inspection_windows"][wid] = {"id": wid, **context, "media": window}
        event(state, "window_inspected", window_id=wid, target_id=target_id)
        return
    require(action == "observe-av", "unknown observation action")
    require(live, "observe-av requires --live and observation transfer policy")
    policy = state.get("observation_policy")
    require(policy and policy.get("active"), "observation is not authorized")
    av.validate_policy(policy)
    if "media_seconds" not in observation["qc"]:
        # Older collected artifacts keep their original immutable media/frames;
        # refresh only technical timing metadata needed by the new adapter.
        observation["qc"] = media.inspect_video(Path(observation["media_path"]))
        require(observation["qc"]["ok"], "source failed refreshed technical QC")
    require(nonempty(state["plan"]["reference_scope_id"]), "observation reference scope missing")
    if target_id == "final":
        imported = [aid for aid in state["accepted"].values() if get_attempt(state, aid).get("imported")]
        generated = any(not get_attempt(state, aid).get("imported") for aid in state["accepted"].values())
        require(set(imported) <= set(policy["imported_target_ids"]), "final contains unauthorized imported media")
        require(not generated or policy["allow_generated_media"], "generated final media transfer not authorized")
    elif target.get("imported"):
        require(target_id in policy["imported_target_ids"], "imported media transfer not authorized")
    else:
        require(policy["allow_generated_media"], "generated media transfer not authorized")
    fps = data.get("fps", 1)
    require(type(fps) in (int, float) and math.isfinite(fps) and 0 < fps <= 8, "FPS must be finite in (0,8]")
    reference_ids = data.get("reference_ids", [])
    require(isinstance(reference_ids, list) and len(set(reference_ids)) == len(reference_ids), "reference_ids must be unique")
    requested_mode = data.get("observation_mode")
    require(requested_mode in (None, "comparison", "blind_facts"), "unknown observation mode")
    client = observer or GeminiObserver(mode=requested_mode or "comparison")
    observer_contract = client.preflight()
    require(isinstance(observer_contract, dict), "observer preflight contract required")
    observation_mode = observer_contract.get("observation_mode", "comparison")
    require(observation_mode in ("comparison", "blind_facts"), "unknown observer contract mode")
    require(requested_mode is None or requested_mode == observation_mode, "observer mode does not match request")
    blind = observation_mode == "blind_facts"
    require(not blind or not reference_ids, "blind facts requires empty reference_ids")
    prompt_version = observer_contract.get("prompt_version", PROMPT_VERSION)
    system_prompt_sha256 = observer_contract.get("system_prompt_sha256", SYSTEM_PROMPT_SHA256)
    require(nonempty(prompt_version) and isinstance(system_prompt_sha256, str) and
            re.fullmatch(r"[0-9a-f]{64}", system_prompt_sha256), "observer prompt identity required")
    require(not blind or ("prompt_version" in observer_contract and "system_prompt_sha256" in observer_contract),
            "blind observer requires explicit prompt identity")
    references = []
    seconds = observation["qc"].get("media_seconds")
    require(positive(seconds), "source media presentation duration is unknown")
    source_media_seconds = seconds
    for eid in reference_ids:
        require(eid in refs and eid in policy["reference_transfer_ids"], "Google reference transfer not authorized")
        ref = refs[eid]
        require(ref.get("verification") == "verified" and ref.get("source_scope") == state["plan"].get("source_scope") and
                ref.get("reference_scope_id") == state["plan"]["reference_scope_id"], "reference not verified in selected scope")
        require(nonempty(ref.get("observation_path")) and nonempty(ref.get("observation_mime_type")), "reference file binding required")
        references.append({"id": eid, "path": ref["observation_path"], "sha256": ref["observation_sha256"],
                           "mime_type": ref["observation_mime_type"], "label": ref["claim"]})
        if ref["observation_mime_type"].startswith(("video/", "audio/")):
            require(positive(ref.get("duration_s")), "reference duration required for observation budget")
            if ref["observation_mime_type"].startswith("video/"):
                actual_ref = media.inspect_video(Path(ref["observation_path"]))
                require(actual_ref["ok"] and abs(actual_ref["duration_s"] - ref["duration_s"]) < 0.15,
                        "reference duration does not match actual media")
                require(positive(actual_ref.get("media_seconds")), "reference media presentation duration unknown")
                seconds += actual_ref["media_seconds"]
            else:
                actual_ref = media.inspect_audio(Path(ref["observation_path"]))
                require(abs(actual_ref["duration_s"] - ref["duration_s"]) < 0.15,
                        "reference duration does not match actual audio")
                seconds += actual_ref["duration_s"]
    window, offset, media_path, uploaded_duration = None, 0.0, observation["media_path"], observation["qc"]["duration_s"]
    has_audio = observation["qc"]["has_audio"]
    if data.get("window_id"):
        item = state.get("inspection_windows", {})[data["window_id"]]
        require(all(item[k] == context[k] for k in context), "window belongs to different media/specification")
        crop = item["media"]
        require(digest(crop["crop"]["path"]) == crop["crop"]["sha256"], "window video changed")
        window = {"start_s": crop["requested_window_s"]["start"], "end_s": crop["requested_window_s"]["end"]}
        offset = crop["crop_to_source_time_mapping"]["source_time_origin_s"]
        media_path = crop["crop"]["path"]
        has_audio = crop["crop"]["inspection"]["has_audio"]
        uploaded_duration = window["end_s"] - window["start_s"]
        crop_seconds = crop["crop"]["inspection"].get("media_seconds")
        require(positive(crop_seconds), "crop media presentation duration is unknown")
        seconds += max(uploaded_duration, crop_seconds) - source_media_seconds
    else:
        require(observation["qc"].get("av_timeline_compatible"),
                "source timeline needs explicit local window normalization before AV observation")
    require(observation["qc"]["video_streams"] == 1 and observation["qc"]["audio_streams"] <= 1,
            "AV observation requires one video and at most one audio stream; select tracks explicitly")
    # Only declared task descriptions cross the API boundary. Internal signed
    # URLs, receipts, local evidence records and generation arguments do not.
    plan = state["plan"]
    visual_only = plan.get("production_contract", {}).get("audio_review") == "disabled"
    if visual_only:
        media_path = str(media.visual_only_copy(Path(media_path), directory / "visual-only"))
        has_audio = False
        for ref in references:
            require(not ref["mime_type"].startswith("audio/"), "audio references excluded from visual review")
            if ref["mime_type"].startswith("video/"):
                ref["source_path"], ref["source_sha256"] = ref["path"], ref["sha256"]
                ref["path"] = str(media.visual_only_copy(Path(ref["path"]), directory / "visual-only"))
                ref["sha256"] = digest(ref["path"])
    spec = {"brief": plan["brief"], "script": plan["script"], "global_constraints": plan.get("global_constraints", []),
            "target": "final" if target_id == "final" else shot_for(state, target["shot_id"])["goal"],
            "required_checks": plan["final_checks"] if target_id == "final" else shot_for(state, target["shot_id"])["required_checks"]}
    if visual_only:
        spec["production_contract"] = plan["production_contract"]
        spec["director_timeline"] = ([get_attempt(state, state["accepted"][s["id"]]).get("director_timeline", s["director_timeline"]) for s in plan["shots"]]
                                    if target_id == "final" else target.get("director_timeline", shot_for(state, target["shot_id"])["director_timeline"]))
    request = {"run_id": state["id"], "target_id": target_id, "source_media_sha256": context["source_hash"],
               "source_duration_s": observation["qc"]["duration_s"], "source_pts_start_s": observation["qc"].get("video_start_s"),
               "source_time_base": observation["qc"].get("video_time_base"), "media_path": media_path,
               "uploaded_media_sha256": digest(media_path), "uploaded_duration_s": uploaded_duration,
               "shot_spec_hash": context["spec_hash"], "reference_hash": context["reference_hash"],
               "references": references, "spec": spec, "requested_fps": fps, "requested_model": DEFAULT_MODEL,
               "prompt_version": prompt_version, "system_prompt_sha256": system_prompt_sha256,
               "observation_mode": observation_mode, "requested_media_resolution": "provider_default",
               "audio_included_in_request": has_audio,
               "audio_review_disabled": visual_only,
               "stream_selection": {"video": observation["qc"].get("video_stream_index"), "audio": "excluded" if visual_only else "all_original"},
               "window": window, "source_offset_s": offset, "qualification_scope": data.get("qualification_scope"),
               "qualification_hash": av.content_hash(policy.get("qualifications", [])),
               "max_output_tokens": policy["max_output_tokens"]}
    request_hash = av.content_hash(request)
    for old in records.values():
        if old["request_hash"] == request_hash and old["status"] == "completed":
            av.verify_artifact(old, **context)
            event(state, "observation_reused", observation_id=old["id"])
            return
    require(not any(a["status"] == "in_progress" for a in records.values()),
            "an interrupted observation must be reconciled with cleanup-observation before another request")
    if any(a["request_hash"] == request_hash for a in records.values()):
        require(nonempty(data.get("retry_reason")), "failed observation retry requires a reason; budget remains charged")
    count = len(records)
    require(count < policy["max_calls"], "observation call budget exhausted")
    if target_id != "final":
        require(count < policy["max_calls"] - policy["reserve_final_calls"], "remaining calls reserved for final review")
    existing = [a for a in records.values() if a["request"]["target_id"] == target_id]
    require(len(existing) < policy["max_calls_per_target"], "target observation budget exhausted")
    if window:
        require(sum(a["request"]["window"] is not None for a in existing) < policy["max_local_calls_per_target"], "local recheck budget exhausted")
    require(sum(a["reserved_media_seconds"] for a in records.values()) + seconds <= policy["max_media_seconds"], "observation media-seconds budget exhausted")
    require(sum(a["request"]["max_output_tokens"] for a in records.values()) + policy["max_output_tokens"] <= policy["max_total_output_tokens"], "observation output-token reservation exhausted")
    if target_id != "final":
        final_count = sum(a["request"]["target_id"] == "final" for a in records.values())
        remaining_final = max(0, policy["reserve_final_calls"] - final_count)
        final_seconds_used = sum(a["reserved_media_seconds"] for a in records.values() if a["request"]["target_id"] == "final")
        require(sum(a["reserved_media_seconds"] for a in records.values()) + seconds <=
                policy["max_media_seconds"] - max(0, policy["reserve_final_media_seconds"] - final_seconds_used),
                "remaining media seconds reserved for final review")
        require(sum(a["request"]["max_output_tokens"] for a in records.values()) + policy["max_output_tokens"] <=
                policy["max_total_output_tokens"] - remaining_final * policy["max_output_tokens"],
                "remaining output tokens reserved for final review")
    capabilities = [] if blind else av.qualified_checks(policy, request, prompt_version)
    for item in [{"path": media_path}, *references]:
        file = Path(item["path"])
        require(file.is_file() and 0 < file.stat().st_size <= 100 * 1024 * 1024, "observation file outside 1 byte..100 MiB cap")
    require(len(references) <= 8, "at most eight observation reference files")
    oid = f"observation_{count + 1:04d}"
    output = directory / oid
    output.mkdir()
    artifact = {"id": oid, "sequence": count + 1, "status": "in_progress", "request": request,
                "request_hash": request_hash, "started_at": now(), "reserved_media_seconds": seconds,
                "remote_files": [], "cleanup_status": "pending", "validated_capabilities": capabilities,
                "actual_sampling_points": None, "estimated_cost": None, "usage": None,
                "qualifications": [] if blind else policy.get("qualifications", []),
                "limitations": ["Sampling points are not disclosed; timestamps are model estimates.",
                                "No automatic lip-sync qualification. Empty capabilities means evaluation pending."]}
    records[oid] = artifact
    event(state, "observation_reserved", observation_id=oid)
    save(path, state)
    def uploaded(record):
        artifact["remote_files"].append(record)
        save(path, state)
    try:
        result = client.observe(request, output, uploaded)
        artifact["result"] = result
        artifact["remote_files"] = result["remote_files"]
        artifact["cleanup_status"] = result["cleanup_status"]
        require(result.get("prompt_version", prompt_version) == prompt_version and
                result.get("system_prompt_sha256", system_prompt_sha256) == system_prompt_sha256 and
                result.get("observation_mode", observation_mode) == observation_mode,
                "observer result prompt identity changed")
        if blind:
            require(result.get("evidence_role") == "blind_facts_only" and
                    result.get("comparison_performed") is False and result.get("qualification_eligible") is False,
                    "blind observation must be facts only without qualification")
            require(not result["response"]["findings"], "blind observation cannot contain comparison findings")
        matching_qualifications = [q for q in ([] if blind else policy.get("qualifications", []))
                                   if q["returned_model"] == result.get("returned_model") and
                                   q["model"] == DEFAULT_MODEL and q["prompt_version"] == prompt_version and q["fps"] == fps]
        artifact["validated_capabilities"] = sorted(set(capabilities) & {c for q in matching_qualifications for c in q["checks"]})
        av.validate_response(result["response"], duration_s=uploaded_duration, has_audio=request["audio_included_in_request"])
        artifact.update(response=result["response"], response_sha256=av.content_hash(result["response"]),
                        source_timeline=av.mapped_response(result["response"], offset),
                        usage=result.get("usage"), status="completed", completed_at=now())
    except Exception as exc:
        error_artifact = getattr(exc, "artifact", {})
        if error_artifact:
            artifact["failure_artifact"] = error_artifact
            artifact["remote_files"] = error_artifact.get("remote_files", artifact["remote_files"])
            artifact["cleanup_status"] = error_artifact.get("cleanup_status", artifact["cleanup_status"])
            artifact["remote_upload_outcome_unknown"] = error_artifact.get("remote_upload_outcome_unknown", False)
        artifact.update(status="failed", error_type=type(exc).__name__, completed_at=now())
        event(state, "observation_failed", observation_id=oid, error_type=type(exc).__name__)
        save(path, state)
        raise
    event(state, "av_observed", observation_id=oid, target_id=target_id)


def task_contract(directory):
    """Bind a managed task's purpose, mode, scope and immutable outer identity."""
    root = directory.parent.parent
    if directory.parent.name != 'video' or not (root / 'task.json').exists():
        return None  # Isolated lower-level contract fixtures have no outer task.
    from ..workspace import load_task
    _, _, record = load_task(root)
    require(record['purpose'] == 'video' and record['video_run_id'] == directory.name,
            'video run does not match its outer task')
    return {'task_id': record['id'], 'mode': record['mode'], 'initial_backend': record['initial_backend'],
            'reference_scope_id': record['reference_scope_id'],
            'metadata_sha256': directing.fingerprint({key: record[key] for key in
                ('id', 'mode', 'initial_backend', 'reference_scope_id', 'video_run_id', 'brief', 'authorization')})}


def enforce_offline(contract, action, *, provider, observer, uploader, image_provider):
    if not contract or contract['mode'] != 'offline':
        return
    selected = {
        'keyframe-generate': image_provider, 'keyframe-upload': uploader, 'upload-handoff': uploader,
        'submit': provider, 'recover': provider, 'poll': provider, 'collect': provider,
        'observe-av': observer, 'cleanup-observation': observer,
    }
    if action in selected:
        client = selected[action]
        require(client is not None and getattr(client, 'production', True) is False,
                'offline task requires an explicit nonproduction provider for this action')


def freeze_evidence(directory, data):
    """Pin managed-task references locally before they can enter a paid request."""
    if not data.get('observation_path'):
        return data
    from ..image.task_store import immutable_write
    source = Path(data['observation_path']).resolve()
    require(source.is_file() and 0 < source.stat().st_size <= 100 * 1024 * 1024,
            'reference file is missing or exceeds the local100MiB limit')
    raw = source.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == data.get('observation_sha256'), 'reference hash changed before freezing')
    suffixes = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp',
                'video/mp4': '.mp4', 'audio/mpeg': '.mp3', 'audio/wav': '.wav'}
    suffix = suffixes.get(data.get('observation_mime_type'))
    require(suffix is not None, 'unsupported reference MIME type')
    root = directory.parent.parent
    from ..workspace import artifact_path
    target = artifact_path(root, 'references/' + ident(data['id']) + suffix)
    target.parent.mkdir(exist_ok=True)
    immutable_write(target, raw, adopt=True)
    return {**data, 'original_local_path': str(source), 'observation_path': str(target)}


def dispatch(action, run_id, data=None, *, live=False, runs=RUNS, provider=None, observer=None, uploader=None, image_provider=None):
    data = json.loads(json.dumps(data or {}, allow_nan=False))
    with locked(run_id, runs, create=action == "init") as (directory, path, state):
        contract = task_contract(directory)
        enforce_offline(contract, action, provider=provider, observer=observer, uploader=uploader, image_provider=image_provider)
        if action == "init":
            require(state is None, "run already exists")
            require(isinstance(data.get("production_contract"), dict), "new runs require production_contract; existing historical runs remain readable")
            require(data["production_contract"].get("version") == 2, "new runs require production_contract version2 with accepted opening frame")
            validate_plan(data)
            if contract:
                require(data["reference_scope_id"] == contract["reference_scope_id"], "video reference scope differs from outer task")
                require(data.get("task_id") == contract["task_id"], "video plan must bind the outer task identity")
            state = {"version": 1, "id": run_id, "plan": data, "authorization": None,
                     "evidence": {}, "attempts": [], "accepted": {}, "final": None, "events": []}
            if contract:
                state["task_contract"] = contract
            if "production_contract" in data:
                state["production_contract_sha256"] = directing.fingerprint(data["production_contract"])
            event(state, "created")
        else:
            require(state is not None, "run has no state")
            directing.verify(state)
            if contract or state.get("task_contract"):
                require(state.get("task_contract") == contract, "outer task contract changed")
        if action == "status":
            return state
        if action in keyframes.ACTIONS:
            keyframes.handle(action, state, directory, path, data, live, image_provider, uploader)
        elif action in OBSERVATION_ACTIONS:
            require(action != "import-video" or not keyframes.enabled(state), "v2 requires generated opening I2V; external video import remains historical-review only")
            handle_observation(action, state, directory, path, data, live, observer)
        elif action == "handoff":
            verify_checkpoints(state)
            target = get_attempt(state, ident(data["attempt_id"]))
            require(target["status"] == "accepted" and state["accepted"].get(target["shot_id"]) == target["id"],
                    "handoff can only export a currently accepted segment")
            _, context, _ = review_context(state, target["id"])
            report = target["reviews"][-1]
            require(report["decision"] == "accept", "handoff needs the current acceptance review")
            hid = f"handoff_{len(state.setdefault('handoffs', {})) + 1:04d}"
            output = Path(tempfile.mkdtemp(prefix=hid + "-", dir=directory))
            package = handoff.export(target["observation"], output, data.get("continuity_state"), data.get("tail_duration_s", 1))
            package.update(id=hid, shot_id=target["shot_id"], **context,
                           accepted_review_sha256=av.content_hash(report), accepted_continuity_state=report["continuity_state"], created_at=now())
            manifest = output / "manifest.json"
            save(manifest, package)
            state["handoffs"][hid] = {**package, "manifest_path": str(manifest), "manifest_sha256": digest(manifest)}
            event(state, "handoff_exported", handoff_id=hid, attempt_id=target["id"])
        elif action == "upload-handoff":
            require(live, "upload-handoff requires --live and explicit generated-handoff transfer authorization")
            auth = state["authorization"]
            require(auth and auth.get("active") and auth.get("allow_generated_handoff_transfer") is True,
                    "generated handoff transfer to fal is not authorized")
            verify_checkpoints(state)
            package = verify_handoff(state, ident(data["handoff_id"]))
            require(not get_attempt(state, package["target_id"]).get("imported"),
                    "generated-handoff authorization does not cover imported source media")
            kind = data.get("artifact")
            require(kind in ("final_frame", "tail_video"), "upload artifact must be final_frame or tail_video")
            artifact = package["artifacts"][kind]
            require(kind != "tail_video" or 2 <= artifact["duration_s"] <= 15,
                    "fal tail-video reference needs at least 2 seconds; export tail_duration_s=2 or use the final frame")
            records = state.setdefault("handoff_uploads", {})
            matching = [record for record in records.values() if record["handoff_id"] == package["id"] and record["artifact"] == kind]
            complete = next((record for record in matching if record["status"] == "completed"), None)
            if complete:
                if complete["evidence_id"] not in auth.setdefault("reference_transfer_ids", []):
                    auth["reference_transfer_ids"].append(complete["evidence_id"])
                event(state, "handoff_upload_reused", upload_id=complete["id"], evidence_id=complete["evidence_id"])
                save(path, state)
                return state
            require(not any(record["status"] == "upload_unknown" for record in records.values()),
                    "an uncertain handoff upload needs provider reconciliation; do not retry automatically")
            if matching:
                require(nonempty(data.get("retry_reason")), "retrying a failed handoff upload requires a reason and remaining budget")
            require(type(auth.get("max_handoff_uploads")) is int and len(records) < auth["max_handoff_uploads"],
                    "handoff upload budget exhausted or missing")
            if uploader is None:
                from .fal_upload import FalMediaUploader
                require(bool(get_setting("FAL_KEY").strip()), "FAL_KEY missing; no upload sent or reserved")
                uploader = FalMediaUploader()
            uploader.preflight()
            uid = f"handoff_upload_{len(records) + 1:04d}"
            eid = f"{uid}_{kind}"
            require(eid not in state["evidence"], "generated upload evidence ID already exists")
            record = {"id": uid, "handoff_id": package["id"], "artifact": kind, "status": "upload_unknown",
                      "source_sha256": artifact["sha256"], "evidence_id": eid, "reserved_at": now()}
            records[uid] = record
            event(state, "handoff_upload_reserved", upload_id=uid)
            save(path, state)
            try:
                result = uploader.upload(Path(artifact["path"]), artifact["mime_type"], expected_sha256=artifact["sha256"])
                require(isinstance(result, dict) and result.get("source_sha256") == artifact["sha256"],
                        "upload receipt does not match the handoff source hash")
                _https_url(result.get("access_url"))
                require(digest(artifact["path"]) == artifact["sha256"], "handoff source changed during upload")
            except Exception as exc:
                record.update(error_type=type(exc).__name__)
                if getattr(exc, "outcome_unknown", True) is False:
                    record["status"] = "failed"
                event(state, "handoff_upload_unresolved", upload_id=uid, error_type=type(exc).__name__)
                save(path, state)
                raise
            evidence = {"id": eid, "reference_scope_id": state["plan"]["reference_scope_id"],
                        "source_scope": state["plan"].get("source_scope"), "kind": "user_reference",
                        "locator": f"handoff:{package['id']}/{kind}",
                        "claim": "Actual ending of accepted segment " + package["shot_id"], "verification": "verified",
                        "verification_method": "reviewed_media_upload", "verification_basis": "artifact_review", "observation": "Source PTS, final-frame inclusion and uploaded bytes are bound to the accepted source media.",
                        "handoff_id": package["id"], "handoff_artifact": kind,
                        "observation_path": artifact["path"], "observation_sha256": artifact["sha256"],
                        "observation_mime_type": artifact["mime_type"], "generation_url": result["access_url"]}
            if kind == "tail_video":
                evidence["duration_s"] = artifact["duration_s"]
            require(eid not in state["evidence"], "generated upload evidence ID already exists")
            state["evidence"][eid] = evidence
            auth.setdefault("reference_transfer_ids", []).append(eid)
            record.update(status="completed", completed_at=now(), result=result)
            event(state, "handoff_uploaded", upload_id=uid, evidence_id=eid)
        elif action == "evidence":
            eid = ident(data["id"])
            require(eid not in state["evidence"], "evidence IDs are immutable; create a new ID")
            require(data.get("reference_scope_id") == state["plan"]["reference_scope_id"], "evidence project mismatch")
            require(data.get("kind") in ("material", "reference_image", "clip", "user_reference"), "unknown evidence kind")
            require(nonempty(data.get("locator")) and nonempty(data.get("claim")), "stable locator and claim required")
            require(data.get("verification") in ("candidate", "verified", "unavailable"), "invalid verification status")
            if data["verification"] == "verified":
                require(nonempty(data.get("observation")) and nonempty(data.get("verification_method")), "verified evidence needs actual source observation")
                require(data.get("verification_basis") in ("direct_observation", "artifact_review"), "metadata alone cannot verify evidence")
            if contract:
                data = freeze_evidence(directory, data)
            state["evidence"][eid] = data
            event(state, "evidence_recorded", evidence_id=eid)
        elif action == "authorize":
            require(nonempty(data.get("user_instruction")), "record the explicit generation instruction")
            require(type(data.get("max_generations")) is int and data["max_generations"] > 0, "max_generations must be a positive integer")
            require(positive(data.get("max_generated_seconds")), "max_generated_seconds required")
            require(type(data.get("max_attempts_per_shot")) is int and data["max_attempts_per_shot"] > 0, "max_attempts_per_shot required")
            if state["plan"].get("production_contract"):
                require(data["max_attempts_per_shot"] <= 5, "current director contract permits at most five attempts per shot")
            require(isinstance(data.get("reference_transfer_ids", []), list), "reference_transfer_ids must be a list")
            require(type(data.get("allow_generated_handoff_transfer", False)) is bool, "allow_generated_handoff_transfer must be boolean")
            if data.get("allow_generated_handoff_transfer") or "max_handoff_uploads" in data:
                require(type(data.get("max_handoff_uploads")) is int and data["max_handoff_uploads"] > 0,
                        "authorized handoff transfer needs a positive max_handoff_uploads")
            state["authorization"] = {**data, "active": True, "at": now()}
            event(state, "authorized", authorization=state["authorization"])
        elif action == "prepare":
            verify_checkpoints(state)
            shot = current_shot(state)
            require(shot and shot["id"] == data.get("shot_id"), "only next unaccepted shot may be prepared")
            previous = last_attempt(state, shot["id"])
            directing.prepare(state, data, previous)
            require(previous is None or previous["status"] in ("rejected", "failed", "superseded"), "resolve current attempt first")
            if previous:
                require(nonempty(data.get("correction")), "retry needs a correction/rationale")
            for eid in shot["evidence_ids"]:
                require(state["evidence"].get(eid, {}).get("verification") == "verified", "shot evidence not verified: " + eid)
            validate_request(data["endpoint"], data["arguments"])
            require(request_duration(data["arguments"]) == shot["duration_s"], "request duration must match planned shot")
            if shot.get("aspect_ratio") and not data["endpoint"].endswith("image-to-video"):
                require(data["arguments"].get("aspect_ratio", "16:9" if data["endpoint"].endswith("text-to-video") else "adaptive") == shot["aspect_ratio"], "request aspect ratio must match planned shot")
            verify_segment_handoff(state, data)
            keyframes.gate(state, data)
            aid = f"attempt_{len(state['attempts']) + 1:04d}"
            attempt = {**data, "id": aid, "status": "prepared", "charged": False}
            state["attempts"].append(attempt)
            event(state, "prepared", attempt_id=aid)
        elif action == "submit":
            require(live, "submit requires --live and recorded explicit user authorization")
            auth = state["authorization"]
            require(auth and auth["active"], "run is not authorized")
            verify_checkpoints(state)
            attempt = get_attempt(state, data["attempt_id"])
            require(attempt["status"] == "prepared", "attempt already submitted or unresolved; never repeat POST")
            shot = current_shot(state)
            require(shot and shot["id"] == attempt["shot_id"], "attempt is no longer current")
            validate_request(attempt["endpoint"], attempt["arguments"])
            verify_segment_handoff(state, attempt)
            keyframes.gate(state, attempt)
            charged = [a for a in state["attempts"] if a["charged"]]
            duration = request_duration(attempt["arguments"])
            require(len(charged) < auth["max_generations"], "generation-count budget exhausted")
            require(sum(request_duration(a["arguments"]) for a in charged) + duration <= auth["max_generated_seconds"], "generated-seconds budget exhausted")
            require(sum(a["shot_id"] == attempt["shot_id"] for a in charged) < auth["max_attempts_per_shot"], "shot attempt budget exhausted")
            # URLs in the actual payload, including nested R2V references, must be bound
            # to explicit per-asset transfer permission. Local/base64 inputs are disallowed.
            def urls(value):
                if isinstance(value, dict):
                    found = []
                    for key, item in value.items():
                        if key.endswith("_urls"):
                            found.extend(item)
                        elif key.endswith("_url"):
                            if item is not None:
                                found.append(item)
                        else:
                            found.extend(urls(item))
                    return found
                if isinstance(value, list):
                    return [u for v in value for u in urls(v)]
                return []
            bindings = attempt.get("reference_bindings", {})
            for url in urls(attempt["arguments"]):
                eid = bindings.get(url)
                require(isinstance(url, str) and url.startswith("https://"), "reference must be HTTPS")
                require(eid in auth.get("reference_transfer_ids", []) and eid in state["evidence"], "reference transfer is not authorized")
                require(state["evidence"][eid].get("verification") == "verified", "transferred reference is not verified")
                require(state["evidence"][eid].get("generation_url") == url, "reference URL is not bound to this evidence; add a fresh evidence ID when a signed URL expires")
            for field in ("reference_video_urls", "reference_audio_urls"):
                durations = [state["evidence"][bindings[url]].get("duration_s") for url in attempt["arguments"].get(field, [])]
                require(all(positive(d) and 2 <= d <= 15 for d in durations) and sum(durations) <= 15, "reference durations must be observed, 2..15 seconds each and <=15 seconds per modality")
            if provider is None:
                key = get_setting("FAL_KEY").strip()
                require(bool(key) and not any(ord(c) < 32 or ord(c) == 127 for c in key), "FAL_KEY missing or invalid; no request sent or budget reserved")
            client = provider or FalProvider()
            # Commit the reservation BEFORE sending. A crash leaves an ambiguous job,
            # which blocks retries until its original receipt has been recovered.
            attempt.update(status="submission_unknown", charged=True)
            event(state, "submission_reserved", attempt_id=attempt["id"])
            save(path, state)
            try:
                receipt = client.submit(attempt["endpoint"], attempt["arguments"])
            except Exception as exc:
                if isinstance(exc, ProviderError) and exc.http_status and 400 <= exc.http_status < 500 and exc.http_status != 408:
                    attempt["status"] = "failed"
                event(state, "submission_unresolved", attempt_id=attempt["id"], error_type=type(exc).__name__)
                save(path, state)
                raise
            attempt.update(status="submitted", receipt=receipt)
            event(state, "submitted", attempt_id=attempt["id"], request_id=receipt["request_id"])
        elif action == "recover":
            attempt = get_attempt(state, data["attempt_id"])
            require(attempt["status"] == "submission_unknown", "only an uncertain submission can be recovered")
            require(nonempty(data.get("recovery_evidence")), "record dashboard/provider evidence tying receipt to original attempt")
            receipt = data["receipt"]
            (provider or FalProvider()).status(receipt)
            attempt.update(status="submitted", receipt=receipt, recovery_evidence=data["recovery_evidence"])
            event(state, "receipt_recovered", attempt_id=attempt["id"])
        elif action == "poll":
            attempt = get_attempt(state, data["attempt_id"])
            require(attempt["status"] in ("submitted", "completed"), "attempt is not a pending remote job")
            client = provider or FalProvider()
            result = client.status(attempt["receipt"])
            attempt["last_status"] = result
            remote = result.get("status")
            if remote == "COMPLETED":
                try:
                    outcome = client.result(attempt["receipt"])
                except ProviderError as exc:
                    if exc.http_status not in (400, 422):
                        raise
                    outcome = {"error": f"Model result HTTP {exc.http_status}"}
                attempt["result"] = outcome
                video = outcome.get("video")
                attempt["status"] = "completed" if not outcome.get("error") and isinstance(video, dict) and nonempty(video.get("url")) else "failed"
            elif remote in ("FAILED", "CANCELLED"):
                attempt["status"] = "failed"
            event(state, "polled", attempt_id=attempt["id"], remote_status=remote)
        elif action == "collect":
            attempt = get_attempt(state, data["attempt_id"])
            require(attempt["status"] == "completed", "generation is not complete")
            output = directory / attempt["id"]
            output.mkdir(exist_ok=True)
            video = output / "video.mp4"
            if not video.exists():
                (provider or FalProvider()).download(attempt["result"]["video"]["url"], video)
            frames = Path(tempfile.mkdtemp(prefix="frames-", dir=output))
            attempt["observation"] = observe(video, frames, shot_for(state, attempt["shot_id"]))
            attempt["status"] = "review_pending" if attempt["observation"]["qc"]["ok"] else "failed"
            event(state, "observed", attempt_id=attempt["id"], technical_pass=attempt["observation"]["qc"]["ok"])
        elif action == "normalize-video":
            verify_checkpoints(state)
            attempt = get_attempt(state, ident(data["attempt_id"]))
            shot = current_shot(state)
            require(shot and shot["id"] == attempt["shot_id"] and last_attempt(state, shot["id"])["id"] == attempt["id"],
                    "normalize only the current attempt of the next unaccepted segment")
            require(attempt["status"] in ("failed", "review_pending") and "observation" in attempt,
                    "normalize only a collected result before review")
            require(not attempt.get("imported") and attempt.get("charged") and attempt.get("receipt", {}).get("request_id")
                    and attempt.get("result", {}).get("video", {}).get("url"), "normalize only this run's collected fal result")
            require(not attempt.get("reviews") and not attempt.get("normalizations"),
                    "reviewed or already normalized media cannot be normalized again")
            for collection in ("av_observations", "human_reviews", "inspection_windows", "handoffs"):
                require(not any(record.get("target_id") == attempt["id"]
                                or record.get("request", {}).get("target_id") == attempt["id"]
                                for record in state.get(collection, {}).values()),
                        "normalization must precede AV observation, inspection, human review and handoff")
            require(nonempty(data.get("reason")), "normalization needs a recorded technical reason")
            verify_segment_handoff(state, attempt)
            original = attempt["observation"]
            verify_observation(original)
            source = Path(original["media_path"])
            require(not source.is_symlink() and source.resolve() == (directory / attempt["id"] / "video.mp4").resolve(),
                    "normalization source must be the original collected file for this attempt")
            output = Path(tempfile.mkdtemp(prefix="normalization-", dir=directory / attempt["id"]))
            record = normalization.normalize(source, output / "video.mp4", duration_s=shot["duration_s"], aspect_ratio=shot.get("aspect_ratio"))
            observed = observe(output / "video.mp4", output / "frames", shot)
            require(observed["qc"]["ok"], "normalized result failed planned shot QC")
            record.update(attempt_id=attempt["id"], shot_id=shot["id"], reason=data["reason"], at=now(),
                          original_observation=original, derived_observation=observed,
                          generation_provenance={key: attempt.get(key) for key in
                                                 ("endpoint", "arguments", "receipt", "result", "reference_bindings", "handoff_id", "charged")})
            manifest = output / "manifest.json"
            save(manifest, record)
            attempt["normalizations"] = [{**record, "manifest_path": str(manifest), "manifest_sha256": digest(manifest)}]
            attempt["observation"] = observed
            attempt["status"] = "review_pending"
            event(state, "video_normalized", attempt_id=attempt["id"], source_sha256=record["source_sha256"],
                  derived_sha256=record["output_sha256"], manifest_path=str(manifest), tail_removed_s=record["tail_removed_s"])
        elif action == "review":
            attempt = get_attempt(state, data["attempt_id"])
            require(attempt["status"] == "review_pending", "attempt is not awaiting review")
            verify_observation(attempt["observation"])
            report = data["report"]
            av_review(state, report, attempt["id"], shot_for(state, attempt["shot_id"])["required_checks"])
            attempt.setdefault("reviews", []).append(report)
            if report["decision"] == "accept":
                attempt["status"] = "accepted"
                state["accepted"][attempt["shot_id"]] = attempt["id"]
            elif report["decision"] == "reject":
                attempt["status"] = "rejected"
            event(state, "reviewed", attempt_id=attempt["id"], decision=report["decision"])
        elif action == "invalidate":
            require(nonempty(data.get("reason")), "invalidation reason required")
            ids = [s["id"] for s in state["plan"]["shots"]]
            start = ids.index(data["shot_id"])
            require(not any(a["status"] in ("submission_unknown", "submitted", "completed", "review_pending") for a in state["attempts"]), "resolve active attempt before invalidating continuity")
            for sid in ids[start:]:
                state["accepted"].pop(sid, None)
                for attempt in state["attempts"]:
                    if attempt["shot_id"] == sid and attempt["status"] in ("accepted", "prepared"):
                        attempt["status"] = "superseded"
            if state["final"]:
                state.setdefault("final_history", []).append(state["final"])
            state["final"] = None
            event(state, "continuity_invalidated", **data)
        elif action == "assemble":
            verify_checkpoints(state)
            require(current_shot(state) is None, "all shots must be accepted before assembly")
            require(state["final"] is None, "final already assembled; review or invalidate it")
            attempts = [get_attempt(state, state["accepted"][s["id"]]) for s in state["plan"]["shots"]]
            for attempt in attempts:
                verify_observation(attempt["observation"])
            final_dir = Path(tempfile.mkdtemp(prefix="final-", dir=directory))
            output = final_dir / "video.mp4"
            media.assemble([Path(a["observation"]["media_path"]) for a in attempts], output)
            boundaries, cuts, elapsed = [], [], 0.0
            for attempt in attempts[:-1]:
                elapsed += attempt["observation"]["qc"]["duration_s"]
                cuts.append(elapsed)
                boundaries.extend([max(0, elapsed - 0.1), elapsed + 0.1])
            state["final"] = {"status": "review_pending", "boundaries_s": cuts,
                              "observation": observe(output, final_dir / "frames", boundary_times=boundaries)}
            event(state, "assembled")
        elif action == "final-review":
            verify_checkpoints(state)
            final = state["final"]
            require(final and final["status"] == "review_pending" and final["observation"]["qc"]["ok"], "final is not awaiting review")
            verify_observation(final["observation"])
            av_review(state, data, "final", state["plan"]["final_checks"])
            final.setdefault("reviews", []).append(data)
            if data["decision"] == "accept":
                final["status"] = "accepted"
                if state["authorization"]:
                    state["authorization"]["active"] = False
                if state.get("observation_policy"):
                    state["observation_policy"]["active"] = False
                if state.get("keyframe_policy"):
                    state["keyframe_policy"]["active"] = False
            event(state, "final_reviewed", decision=data["decision"])
        elif action == "stop":
            if state["authorization"]:
                state["authorization"]["active"] = False
            if state.get("observation_policy"):
                state["observation_policy"]["active"] = False
            if state.get("keyframe_policy"):
                state["keyframe_policy"]["active"] = False
            event(state, "stopped")
            save(path, state)
            # Local stop is durable even if provider cancellation fails.
            if data.get("attempt_id"):
                attempt = get_attempt(state, data["attempt_id"])
                require("receipt" in attempt, "no receipt: cancellation needs provider reconciliation")
                enforce_offline(contract, "submit", provider=provider, observer=observer, uploader=uploader, image_provider=image_provider)
                attempt["cancel_response"] = (provider or FalProvider()).cancel(attempt["receipt"])
                event(state, "cancel_requested", attempt_id=attempt["id"])
        elif action != "init":
            raise ValueError("unknown action: " + action)
        save(path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["doctor", "init", "status", "evidence", "authorize", "handoff", "upload-handoff", "prepare", "submit", "recover", "poll", "collect", "normalize-video", "review", "invalidate", "assemble", "final-review", "stop", *sorted(OBSERVATION_ACTIONS | keyframes.ACTIONS)])
    parser.add_argument("--run")
    parser.add_argument("--runs", type=Path, help="Explicit task-local video directory")
    parser.add_argument("--input", type=Path, help="UTF-8 JSON input; never supply API keys")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "doctor":
            result = {"python": sys.version.split()[0], "fal_key_present": bool(get_setting("FAL_KEY")),
                      "gemini_key_present": bool(get_setting("GEMINI_API_KEY")),
                      "openai_key_present": bool(get_setting("OPENAI_API_KEY")),
                      "image_model_configured": bool(get_setting("OPENAI_IMAGE_MODEL")),
                      "ffmpeg": bool(shutil.which("ffmpeg")), "ffprobe": bool(shutil.which("ffprobe")),
                      "references": "Supply verified local evidence through the task workflow."}
        else:
            require(args.run, "--run required")
            data = json.loads(args.input.read_text()) if args.input else {}
            result = dispatch(args.action, args.run, data, live=args.live, runs=args.runs)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as exc:
        # Do not print provider response bodies, prompts or secret-bearing URLs.
        safe = str(exc) if isinstance(exc, (ValueError, StopIteration)) else type(exc).__name__
        print(json.dumps({"ok": False, "error": safe or "record not found"}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
