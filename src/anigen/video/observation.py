"""Observation contracts. A model report is evidence, never a review decision."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

VIDEO_CHECKS = {"temporal", "motion", "action", "action_timing", "camera", "pacing", "narrative"}
AUDIO_CHECKS = {"audio", "dialogue", "av_sync"}
DYNAMIC_CHECKS = VIDEO_CHECKS | AUDIO_CHECKS | {"lip_sync"}
STATIC_CHECKS = {"identity", "appearance", "composition", "scene", "prompt_adherence",
                 "continuity", "intent", "visual_artifacts"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_policy(policy):
    require(text(policy.get("user_instruction")), "record explicit observation and Google transfer authorization")
    require(policy.get("review_authority", "qualified_observer") in
            ("qualified_observer", "codex_director"), "invalid review authority")
    require(policy.get("allow_task_spec") is True, "task-spec transfer must be authorized")
    require(type(policy.get("allow_generated_media")) is bool, "allow_generated_media must be explicit")
    for field in ("reference_transfer_ids", "imported_target_ids"):
        value = policy.get(field)
        require(isinstance(value, list) and all(text(x) for x in value), field + " must be a list of IDs")
    defaults = {"max_calls_per_target": 3, "max_local_calls_per_target": 1,
                "reserve_final_calls": 1, "max_output_tokens": 4096}
    for key, value in defaults.items():
        policy.setdefault(key, value)
    for key in ("max_calls", "max_calls_per_target", "max_output_tokens"):
        require(type(policy.get(key)) is int and policy[key] > 0, key + " must be a positive integer")
    for key in ("max_local_calls_per_target", "reserve_final_calls"):
        require(type(policy.get(key)) is int and policy[key] >= 0, key + " must be a nonnegative integer")
    require(policy["reserve_final_calls"] < policy["max_calls"], "reserve must leave calls for shots")
    require(policy["max_output_tokens"] <= 4096, "max_output_tokens exceeds adapter cap")
    require(number(policy.get("max_media_seconds")) and policy["max_media_seconds"] > 0,
            "max_media_seconds must be finite and positive")
    policy.setdefault("reserve_final_media_seconds", 0)
    require(number(policy["reserve_final_media_seconds"]) and 0 <= policy["reserve_final_media_seconds"] < policy["max_media_seconds"],
            "final-media reservation must be nonnegative and leave room for shots")
    policy.setdefault("max_total_output_tokens", policy["max_calls"] * policy["max_output_tokens"])
    require(type(policy["max_total_output_tokens"]) is int and policy["max_total_output_tokens"] > 0,
            "max_total_output_tokens must be positive")
    policy.setdefault("qualifications", [])
    require(isinstance(policy["qualifications"], list), "qualifications must be a list")
    for qualification in policy["qualifications"]:
        validate_qualification(qualification)
    return policy


def validate_qualification(value):
    """Human-approved evaluation scope; never infer this from schema-valid output."""
    require(text(value.get("user_confirmation")) and text(value.get("scope")),
            "qualification needs explicit user confirmation and evaluated scope")
    require(text(value.get("model")) and text(value.get("returned_model")) and text(value.get("prompt_version")), "qualification requested/returned model and prompt required")
    require(number(value.get("fps")) and value["fps"] > 0, "qualification FPS required")
    checks = value.get("checks")
    require(isinstance(checks, list) and checks and set(checks) <= VIDEO_CHECKS | AUDIO_CHECKS,
            "qualification contains unsupported checks; lip_sync requires human confirmation")
    require(text(value.get("evaluation_report_path")), "qualification evaluation report required")
    require(file_hash(value["evaluation_report_path"]) == value.get("evaluation_report_sha256"),
            "qualification evaluation report changed")


def qualified_checks(policy, request, prompt_version):
    checks = set()
    for qualification in policy.get("qualifications", []):
        validate_qualification(qualification)
        if (qualification["model"] == request["requested_model"] and qualification["scope"] == request.get("qualification_scope") and
                qualification["prompt_version"] == prompt_version and qualification["fps"] == request["requested_fps"]):
            # The caller must use a qualification whose stated content domain applies.
            require(request.get("qualification_scope") == qualification["scope"],
                    "observation must explicitly match evaluated qualification scope")
            checks.update(qualification["checks"])
    if not request["audio_included_in_request"]:
        checks -= AUDIO_CHECKS
    return sorted(checks)


def validate_response(response, *, duration_s, has_audio):
    require(isinstance(response, dict), "observation response must be an object")
    fields = ("events", "dialogue_segments", "audio_events", "findings", "uncertainties")
    require(set(response) == set(fields), "observation response has missing or unexpected fields")
    require(any(response.values()), "empty observation cannot provide evidence")
    for field in fields:
        rows = response[field]
        require(isinstance(rows, list) and len(rows) <= 500, "invalid observation array: " + field)
        for row in rows:
            require(isinstance(row, dict), "observation entry must be an object")
            start, end = row.get("start_s"), row.get("end_s")
            require(number(start) and number(end) and 0 <= start <= end <= duration_s,
                    "observation timestamps outside supplied video")
            if field == "dialogue_segments":
                expected = {"start_s", "end_s", "text", "language", "speaker", "uncertainty"}
                require(text(row.get("text")), "heard dialogue text required")
                require(all(isinstance(row.get(k), str) for k in ("language", "speaker", "uncertainty")),
                        "dialogue details must be strings")
            else:
                expected = {"start_s", "end_s", "description", "evidence"}
                require(text(row.get("description")) and text(row.get("evidence")), "observation fact and evidence required")
            if field == "findings":
                expected |= {"check", "status"}
                require(row.get("check") in STATIC_CHECKS | DYNAMIC_CHECKS, "unknown finding check")
                require(row.get("status") in ("observed", "fail", "unknown"), "invalid finding status")
                if row["check"] in AUDIO_CHECKS | {"lip_sync"} and not has_audio:
                    require(row["status"] != "observed", "absent audio cannot support positive audio evidence")
            require(set(row) == expected, "observation entry has missing or unexpected fields")
    require(has_audio or not response["audio_events"] and not response["dialogue_segments"],
            "observer claimed to hear audio in a video without audio")
    return response


def mapped_response(response, source_offset_s):
    return {key: [{**row, "source_start_s": row["start_s"] + source_offset_s,
                   "source_end_s": row["end_s"] + source_offset_s,
                   "time_precision": "model_estimate"} for row in rows]
            for key, rows in response.items()}


def verify_artifact(artifact, *, source_hash, spec_hash, reference_hash, target_id, run_id):
    require(artifact.get("status") == "completed", "observation is not complete")
    request = artifact["request"]
    require(request["run_id"] == run_id and request["target_id"] == target_id,
            "observation belongs to another run or target")
    require(request["source_media_sha256"] == source_hash, "observation belongs to different media")
    require(request["shot_spec_hash"] == spec_hash, "observation belongs to an older specification")
    require(request["reference_hash"] == reference_hash, "observation references changed")
    require(artifact["request_hash"] == content_hash(request), "observation request changed")
    require(file_hash(request["media_path"]) == request["uploaded_media_sha256"], "uploaded media changed")
    for reference in request["references"]:
        require(file_hash(reference["path"]) == reference["sha256"], "uploaded reference changed")
    result = artifact["result"]
    require(file_hash(result["raw_response_path"]) == result["raw_response_sha256"], "raw response changed")
    require(content_hash(artifact["response"]) == artifact["response_sha256"], "parsed observation changed")
    validate_response(artifact["response"], duration_s=request["uploaded_duration_s"],
                      has_audio=request["audio_included_in_request"])
    for qualification in artifact.get("qualifications", []):
        validate_qualification(qualification)
    return artifact


def resolve_ref(reference, artifacts):
    require(isinstance(reference, str), "evidence reference must be text")
    match = re.fullmatch(r"([A-Za-z0-9_-]+)/(events|dialogue_segments|audio_events|findings|uncertainties)/(\d+)", reference)
    require(match is not None, "use observation_id/collection/index evidence references")
    oid, collection, index = match.groups()
    require(oid in artifacts, "evidence points to an unlisted observation")
    rows = artifacts[oid]["response"][collection]
    require(int(index) < len(rows), "evidence index outside observation")
    return artifacts[oid], collection, rows[int(index)]


def director_resolution(resolution, chosen, name, original, finding):
    if "by_check" in resolution:
        require(isinstance(resolution["by_check"], dict) and name in resolution["by_check"],
                "director uncertainty resolution missing this check")
        resolution = resolution["by_check"][name]
    require(text(resolution.get("reason")) and resolution.get("evidence_refs"),
            "director conflict resolution needs specific evidence")
    start = finding["start_s"] + original["request"].get("source_offset_s", 0)
    end = finding["end_s"] + original["request"].get("source_offset_s", 0)
    intervals = []
    for ref in resolution["evidence_refs"]:
        artifact, collection, row = resolve_ref(ref, chosen)
        require(collection != "uncertainties" and row.get("status", "observed") == "observed",
                "director conflict resolution cites adverse evidence")
        if collection == "findings":
            require(row["check"] == name, "director resolution uses unrelated check")
        elif name == "dialogue":
            require(collection == "dialogue_segments", "dialogue resolution needs speech evidence")
        elif name in AUDIO_CHECKS:
            require(collection == "audio_events" or name == "av_sync" and collection == "events",
                    "audio resolution needs audio evidence")
        else:
            require(collection == "events", "visual resolution needs visual events")
        offset = artifact["request"].get("source_offset_s", 0)
        intervals.append((row["start_s"] + offset, row["end_s"] + offset))
    covered_until = start
    reached = False
    for left, right in sorted(intervals):
        if right < covered_until:
            continue
        if left > covered_until:
            break
        reached = True
        covered_until = max(covered_until, right)
    require(reached and covered_until >= end, "director resolution does not cover conflicting time range")


def verify_dynamic_passes(report, artifacts, human_records, context, policy=None):
    """Bind evidence; distinguish director judgment from qualified automation."""
    director = report.get("review_authority") == "codex_director"
    if director:
        require((policy or {}).get("review_authority") == "codex_director",
                "director review is not enabled by the recorded policy")
        review = report.get("director_review", {})
        require(review.get("reviewer") == "codex" and text(review.get("rationale"))
                and isinstance(review.get("limitations"), list), "director review needs rationale and limitations")
    selected = report.get("observation_ids", [])
    require(isinstance(selected, list) and len(set(selected)) == len(selected), "observation_ids must be unique")
    chosen = {}
    for oid in selected:
        require(oid in artifacts, "unknown observation ID")
        artifact = verify_artifact(artifacts[oid], **context)
        current = []
        for qualification in (policy or {}).get("qualifications", []):
            validate_qualification(qualification)
            request = artifact["request"]
            if (qualification["model"] == request["requested_model"] and
                    qualification["returned_model"] == artifact["result"].get("returned_model") and
                    qualification["fps"] == request["requested_fps"] and
                    qualification["prompt_version"] == request["prompt_version"] and
                    qualification["scope"] == request.get("qualification_scope")):
                current.extend(qualification["checks"])
        chosen[oid] = {**artifact, "validated_capabilities": set(artifact["validated_capabilities"]) & set(current)}
    if director:
        require(any(a["request"].get("observation_mode", "comparison") == "comparison"
                    and a["request"]["window"] is None for a in chosen.values()),
                "director review requires complete full-video comparison")
        for oid, artifact in artifacts.items():
            req = artifact.get("request", {})
            if not (artifact.get("status") == "completed" and
                    all(req.get(k) == context[v] for k, v in
                        (("source_media_sha256", "source_hash"), ("shot_spec_hash", "spec_hash"),
                         ("reference_hash", "reference_hash"), ("target_id", "target_id"), ("run_id", "run_id")))):
                continue
            for index, finding in enumerate(artifact["response"]["findings"]):
                name = finding["check"]
                if name in STATIC_CHECKS and finding["status"] in ("fail", "unknown") and report.get("checks", {}).get(name, {}).get("status") == "pass":
                    key = f"{oid}/findings/{index}"
                    resolution = report.get("director_conflict_resolutions", {}).get(key)
                    require(isinstance(resolution, dict), "unresolved static observer conflict")
                    director_resolution(resolution, chosen, name, artifact, finding)
    for name, check in report.get("checks", {}).items():
        if name not in DYNAMIC_CHECKS or check.get("status") != "pass":
            continue
        human_id = check.get("human_review_id")
        if human_id:
            require(human_id in human_records, "unknown human confirmation")
            human = human_records[human_id]
            require(all(human.get(k) == context[k] for k in ("source_hash", "spec_hash", "reference_hash", "target_id", "run_id")),
                    "human confirmation belongs to old media, scope or target")
            require(name in human["checks"], "human confirmation does not cover this check")
            continue
        require(name != "lip_sync", "precise lip_sync requires explicit human confirmation")
        if director:
            require(text(check.get("director_reason")), "director pass needs per-check reasoning")
        refs = check.get("evidence_refs", [])
        require(isinstance(refs, list) and refs, "dynamic pass needs actual observation evidence references")
        supporting = []
        for ref in refs:
            artifact, collection, row = resolve_ref(ref, chosen)
            if not director:
                require(name in artifact["validated_capabilities"], "observer has no evaluated qualification for check")
            require(collection != "uncertainties", "uncertainty cannot support a pass")
            if collection == "findings":
                require(row["check"] == name and row["status"] == "observed", "finding does not support this pass")
            elif name == "dialogue":
                require(collection == "dialogue_segments", "dialogue needs actual heard speech")
            elif name in AUDIO_CHECKS:
                require(collection == "audio_events" or name == "av_sync" and collection == "events",
                        "audio check needs actual audio evidence")
            else:
                require(collection == "events", "video check needs observed events")
            supporting.append(artifact)
        require(any(a["request"]["window"] is None for a in supporting), "local window cannot clear full-video review")
        if name in AUDIO_CHECKS:
            require(all(a["request"]["audio_included_in_request"] for a in supporting), "audio not included")
        # All current successful observations count, including inconvenient reports
        # omitted from observation_ids. Corrections must cite later local evidence.
        conflicts = []
        for oid, artifact in artifacts.items():
            req = artifact.get("request", {})
            if (artifact.get("status") == "completed" and req.get("source_media_sha256") == context["source_hash"]
                    and req.get("shot_spec_hash") == context["spec_hash"] and req.get("reference_hash") == context["reference_hash"]
                    and req.get("target_id") == context["target_id"]):
                for index, finding in enumerate(artifact["response"]["findings"]):
                    if finding["check"] == name and finding["status"] in ("fail", "unknown"):
                        conflicts.append(f"{oid}/findings/{index}")
                # The response schema does not assign uncertainty to individual
                # checks, so conservatively require all of it to be resolved.
                conflicts.extend(f"{oid}/uncertainties/{index}" for index, _ in enumerate(artifact["response"]["uncertainties"]))
        resolutions = check.get("resolutions", {})
        require(isinstance(resolutions, dict), "resolutions must map conflicting evidence to follow-up evidence")
        for conflict in conflicts:
            if director and conflict in report.get("director_conflict_resolutions", {}):
                resolution = report["director_conflict_resolutions"][conflict]
                original, _, finding = resolve_ref(conflict, artifacts)
                director_resolution(resolution, chosen, name, original, finding)
                continue
            require(conflict in resolutions, "unresolved observer failure or unknown: " + conflict)
            original, _, finding = resolve_ref(conflict, artifacts)
            followup, _, resolved = resolve_ref(resolutions[conflict], chosen)
            require(followup["sequence"] > original["sequence"] and
                    resolutions[conflict] in refs and resolved.get("status", "observed") == "observed",
                    "conflict needs later supporting observation")
            offset = original["request"].get("source_offset_s", 0)
            follow_window = followup["request"]["window"]
            if follow_window:
                require(follow_window["start_s"] <= finding["start_s"] + offset and
                        follow_window["end_s"] >= finding["end_s"] + offset,
                        "follow-up window does not cover conflicting finding")
