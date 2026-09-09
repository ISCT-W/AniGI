"""Offline paired AV fixtures and evidence-bound scoring. Never calls a model/API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
from typing import Any

VERSION = 1
FPS = 30
DURATION_MS = 5000
ARMS = {
    "A": {"name": "sparse_frames", "fps": 1, "audio": False, "local_review": False},
    "B": {"name": "av_1fps", "fps": 1, "audio": True, "local_review": False},
    "C": {"name": "av_1fps_plus_local", "fps": 1, "audio": True, "local_review": True},
    "D": {"name": "av_4fps", "fps": 4, "audio": True, "local_review": False},
}
LIMITATIONS = [
    "Synthetic colored shapes and tones do not establish character, Japanese dialogue, or lip-sync quality.",
    "Ground truth describes the synthetic stimulus; event localization has video-frame and audio-codec uncertainty.",
    "Requested model FPS does not prove which sampling points the provider actually observed.",
    "Preparing media or scoring supplied judgments does not execute or validate any observation model.",
]


class EvaluationError(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _number(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise EvaluationError(f"{field} must be a finite nonnegative number")
    return value


def _intervals(value: Any, duration: float, field: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise EvaluationError(f"{field} must be an array")
    for interval in value:
        if not isinstance(interval, list) or len(interval) != 2:
            raise EvaluationError(f"{field} entries must be [start_ms, end_ms]")
        start, end = (_number(interval[0], field), _number(interval[1], field))
        if not start < end <= duration:
            raise EvaluationError(f"{field} interval outside media or empty")
    return value


def _moving(time_expr: str = "t") -> str:
    # overlay evaluates x per frame, unlike drawbox's x expression.
    return f"[0:v][2:v]overlay=x='16+40*({time_expr})':y=70:shortest=1[v]"


def _beep(start: float = 2.0) -> str:
    return f"aevalsrc=if(between(t\\,{start}\\,{start + .1})\\,0.4*sin(2*PI*880*t)\\,0):s=48000:d=5"


def _cases() -> list[dict]:
    """Each spec is shared by its control and single-factor variant."""
    normal = _moving()
    tone = "sine=frequency=440:sample_rate=48000:duration=5"
    silence = "anullsrc=r=48000:cl=mono:d=5"
    impact = "[0:v]drawbox=x=120:y=50:w=80:h=80:color=white:t=fill:enable='gte(n,60)*lt(n,63)'[v]"
    cases = []

    def add(name, check, requirement, variant_video, interval, *, control_video=normal,
            control_audio=tone, variant_audio=None, acceptable=False, factor=None):
        cases.append({"pair_id": name, "check": check, "requirement": requirement,
                      "control_video": control_video, "variant_video": variant_video,
                      "control_audio": control_audio, "variant_audio": variant_audio or control_audio,
                      "interval_ms": interval, "variant_acceptable": acceptable,
                      "changed_factor": factor or name})

    order = "[0:v]drawbox=x=100:y=50:w=80:h=80:color={}:t=fill:enable='lt(n,75)',drawbox=x=100:y=50:w=80:h=80:color={}:t=fill:enable='gte(n,75)'[v]"
    add("action_order", "action", "A red square is followed by a blue square at 2.5 s.",
        order.format("blue", "red"), [0, 5000], control_video=order.format("red", "blue"))
    add("action_incomplete", "action", "The white square moves continuously from x=16 to about x=215 across 5 s.",
        _moving("min(t,2)"), [2000, 5000])
    add("unexpected_freeze", "motion", "The white square moves continuously with no pause.",
        _moving("if(lt(t,2),t,if(lt(t,3),2,t-1))"), [2000, 3000])
    for count in (1, 3, 6):
        video = normal.replace("[v]", "[moving]") + f";[moving]drawbox=x=120:y=30:w=70:h=60:color=magenta:t=fill:enable='gte(n,37)*lt(n,{37+count})'[v]"
        add(f"gap_defect_{count}_frames", "visual_artifacts", "Only the white moving square is present; no magenta block appears.",
            video, [37000/FPS, (37+count)*1000/FPS], factor=f"insert_{count}_magenta_frames")
    add("unexpected_silence", "audio", "A continuous audible tone is required for all 5 s.",
        normal, [0, 5000], variant_audio=silence, factor="replace_tone_with_silence")
    for offset in (-1, -.5, -.2, .2, .5, 1):
        label = ("early" if offset < 0 else "late") + f"_{round(abs(offset)*1000)}ms"
        add(f"impact_{label}", "audio", "The brief visible impact and tone start together at 2 s; allowed error is 0.1 s.",
            impact, [min(2, 2+offset)*1000, (max(2, 2+offset)+.1)*1000],
            control_video=impact, control_audio=_beep(), variant_audio=_beep(2+offset),
            factor=f"audio_offset_{offset:+g}_seconds")
    add("cut_black_gap", "continuity", "The square keeps moving through the midpoint without a black interruption.",
        normal.replace("[v]", "[moving]") + ";[moving]drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='gte(n,75)*lt(n,81)'[v]",
        [2500, 2700], factor="insert_black_at_cut")
    add("cut_position_jump", "continuity", "The square keeps moving at constant speed through the midpoint without jumping backwards.",
        _moving("if(lt(t,2.5),t,t-1.25)"), [2500, 2600], factor="reset_position_at_cut")
    black = normal.replace("[v]", "[moving]") + ";[moving]drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='gte(n,60)*lt(n,66)'[v]"
    add("allowed_black", "continuity", "A moving square may include a deliberate black transition from 2.0 to 2.2 s.",
        black, [2000, 2200], acceptable=True, factor="optional_designed_black_transition")
    add("allowed_hold", "motion", "A square moving right may include a deliberate animation hold from 2 to 3 s.",
        _moving("if(lt(t,2),t,if(lt(t,3),2,t-1))"), [2000, 3000], acceptable=True,
        factor="optional_designed_animation_hold")
    add("allowed_silence", "audio", "The moving square can have either a background tone or a fully silent audio track; neither is a defect.",
        normal, [0, 5000], variant_audio=silence, acceptable=True, factor="optional_silent_sound_design")
    return cases


def prepare(output_dir: str | Path) -> dict:
    """Create 18 fixed pairs in a NEW directory. Existing outputs are never overwritten."""
    root = Path(output_dir).resolve()
    if root.exists():
        raise EvaluationError("prepare requires a new output directory")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise EvaluationError("FFmpeg and ffprobe are required")
    root.mkdir(parents=True, mode=0o700)
    media_dir = root / "media"
    media_dir.mkdir()
    manifest = {"schema_version": VERSION, "benchmark": "synthetic_av_pairs_v1", "arms": ARMS,
                "split_policy": "Pairs stay together; fixed alternating pair indices: tune, holdout.",
                "limitations": LIMITATIONS, "cases": []}
    try:
        for index, pair in enumerate(_cases()):
            split = "tune" if index % 2 == 0 else "holdout"
            spec = {"duration_ms": DURATION_MS, "fps": FPS, "width": 320, "height": 180,
                    "check": pair["check"], "requirement": pair["requirement"]}
            for member in ("control", "variant"):
                case_id = f"{pair['pair_id']}_{member}"
                path = media_dir / f"{case_id}.mp4"
                command = [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-n",
                           "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=5",
                           "-f", "lavfi", "-i", pair[f"{member}_audio"],
                           "-f", "lavfi", "-i", "color=c=white:s=24x24:r=30:d=5",
                           "-filter_complex", pair[f"{member}_video"], "-map", "[v]", "-map", "1:a:0",
                           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-t", "5", "-movflags", "+faststart", str(path)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=45)
                if result.returncode:
                    raise EvaluationError(f"FFmpeg failed for {case_id}: {result.stderr[-2000:]}")
                probe_result = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams",
                                               "-show_format", "-of", "json", str(path)],
                                              capture_output=True, text=True, timeout=30)
                if probe_result.returncode:
                    raise EvaluationError(f"ffprobe failed for {case_id}")
                probe = json.loads(probe_result.stdout)
                videos = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
                audios = [stream for stream in probe["streams"] if stream["codec_type"] == "audio"]
                if len(videos) != 1 or len(audios) != 1 or int(videos[0]["nb_read_frames"]) != 150:
                    raise EvaluationError(f"Unexpected streams/frame count for {case_id}")
                if abs(float(probe["format"]["duration"]) - 5) > .05:
                    raise EvaluationError(f"Unexpected duration for {case_id}")
                valid = member == "control" or pair["variant_acceptable"]
                manifest["cases"].append({"case_id": case_id, "pair_id": pair["pair_id"], "member": member,
                    "split": split, "control_case_id": pair["pair_id"] + "_control",
                    "media_path": str(path.relative_to(root)), "media_sha256": _file_digest(path),
                    "spec": spec, "spec_sha256": _digest(spec), "changed_factor": pair["changed_factor"],
                    "ground_truth": {"expected_accept": valid, "severity": "none" if valid else "severe",
                        "defect_intervals_ms": [] if valid else [pair["interval_ms"]],
                        "designed_event_intervals_ms": [pair["interval_ms"]],
                        "local_time_origin_ms": 0, "annotation_source": "deterministic_fixture_recipe",
                        "video_quantization_ms": 1000/FPS, "audio_codec_timing_tolerance_ms": 50},
                    "recipe": {"video_filter": pair[f"{member}_video"], "audio_source": pair[f"{member}_audio"]}})
        _write(root / "manifest.json", manifest)
        # No prefilled decisions: this template cannot be mistaken for observed results.
        _write(root / "judgments.template.json", {"schema_version": VERSION, "arm_configs": {}, "judgments": []})
        return {"manifest": str(root / "manifest.json"), "pairs": len(_cases()),
                "clips": len(manifest["cases"]), "status": "synthetic_media_prepared",
                "model_calls": 0, "limitations": LIMITATIONS}
    except Exception:
        # Preserve partial media for diagnosis; never claim the benchmark is ready.
        _write(root / "PREPARATION_FAILED.json", {"status": "failed", "completed_cases": len(manifest["cases"])})
        raise


def _load(path: Path) -> dict:
    def reject_constant(value):
        raise EvaluationError(f"Nonfinite JSON number: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Cannot read JSON: {path.name}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != VERSION:
        raise EvaluationError("Unsupported or missing schema_version")
    return value


def _validate_manifest(manifest: dict, root: Path) -> dict[str, dict]:
    entries = manifest.get("cases")
    if not isinstance(entries, list) or not entries:
        raise EvaluationError("Manifest needs nonempty cases")
    cases, pairs = {}, {}
    for case in entries:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"]:
            raise EvaluationError("Case requires case_id")
        identity = case["case_id"]
        if identity in cases:
            raise EvaluationError("Duplicate case_id")
        if case.get("split") not in ("tune", "holdout"):
            raise EvaluationError("Unknown split")
        if case.get("member") not in ("control", "variant") or not isinstance(case.get("pair_id"), str):
            raise EvaluationError("Case requires pair_id and member")
        pair_cases = pairs.setdefault(case["pair_id"], [])
        if pair_cases and (case["split"] != pair_cases[0]["split"] or case.get("spec_sha256") != pair_cases[0].get("spec_sha256")):
            raise EvaluationError("Control pair crosses splits or changes specification")
        spec = case.get("spec")
        if not isinstance(spec, dict) or _digest(spec) != case.get("spec_sha256"):
            raise EvaluationError("Case specification hash mismatch")
        duration = _number(spec.get("duration_ms"), "duration_ms")
        if duration == 0 or not isinstance(spec.get("check"), str) or not spec["check"]:
            raise EvaluationError("Case requires positive duration and check")
        truth = case.get("ground_truth", {})
        if not isinstance(truth, dict) or type(truth.get("expected_accept")) is not bool:
            raise EvaluationError("Ground truth requires expected_accept")
        if truth.get("severity") not in ("none", "severe", "minor"):
            raise EvaluationError("Unknown severity")
        intervals = _intervals(truth.get("defect_intervals_ms"), duration, "ground truth")
        if truth["expected_accept"] and (intervals or truth["severity"] != "none"):
            raise EvaluationError("Valid case cannot carry a defect")
        if not truth["expected_accept"] and (not intervals or truth["severity"] == "none"):
            raise EvaluationError("Defect case requires intervals and severity")
        if case["member"] == "control" and not truth["expected_accept"]:
            raise EvaluationError("Control must satisfy its specification")
        if not isinstance(case.get("media_path"), str):
            raise EvaluationError("Case requires media_path")
        media = (root / case["media_path"]).resolve()
        if not media.is_relative_to(root.resolve()) or not media.is_file() or _file_digest(media) != case.get("media_sha256"):
            raise EvaluationError("Missing, changed, or out-of-directory case media")
        pair_cases.append(case)
        cases[identity] = case
    for members in pairs.values():
        if len(members) != 2 or {case["member"] for case in members} != {"control", "variant"}:
            raise EvaluationError("Each pair requires exactly one control and variant")
        control = next(case["case_id"] for case in members if case["member"] == "control")
        if any(case.get("control_case_id") != control for case in members):
            raise EvaluationError("Wrong control_case_id")
    return cases


def _validate_judgments(document: dict, cases: dict) -> dict[tuple[str, str], dict]:
    rows = document.get("judgments")
    if not isinstance(rows, list):
        raise EvaluationError("judgments must be an array")
    indexed = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("case_id"), str)
                or not isinstance(row.get("arm"), str)
                or row["case_id"] not in cases or row["arm"] not in ARMS):
            raise EvaluationError("Unknown judgment case or arm")
        case = cases[row["case_id"]]
        key = (row["arm"], row["case_id"])
        if key in indexed:
            raise EvaluationError("Duplicate judgment for case and arm")
        if "split" in row and row["split"] != case["split"]:
            raise EvaluationError("Judgment split does not match fixed manifest split")
        if row.get("media_sha256") != case["media_sha256"] or row.get("spec_sha256") != case["spec_sha256"]:
            raise EvaluationError("Judgment media/spec hash mismatch")
        if row.get("decision") not in ("accept", "reject", "unknown", "needs_review"):
            raise EvaluationError("Decision must be accept/reject/unknown/needs_review")
        _intervals(row.get("finding_intervals_ms", []), case["spec"]["duration_ms"], "finding")
        usage = row.get("usage", {})
        if not isinstance(usage, dict):
            raise EvaluationError("usage must be an object")
        for field in ("calls", "cost_usd", "latency_ms"):
            number = _number(usage.get(field), field, nullable=True)
            if field == "calls" and number is not None and int(number) != number:
                raise EvaluationError("calls must be integral")
        indexed[key] = row
    return indexed


def _ratio(count: int, denominator: int) -> dict:
    return {"count": count, "denominator": denominator, "rate": count / denominator if denominator else None}


def _metrics(cases: list[dict], rows: dict, arm: str) -> dict:
    available = [(case, rows[(arm, case["case_id"])]) for case in cases if (arm, case["case_id"]) in rows]
    missing = [case["case_id"] for case in cases if (arm, case["case_id"]) not in rows]
    severe = [(c, r) for c, r in available if c["ground_truth"]["severity"] == "severe"]
    valid = [(c, r) for c, r in available if c["ground_truth"]["expected_accept"]]
    unknown = sum(r["decision"] in ("unknown", "needs_review") for _, r in available)
    errors, located_cases, unlocalized_rejects = [], 0, 0
    for case, row in available:
        truth = case["ground_truth"]["defect_intervals_ms"]
        findings = row.get("finding_intervals_ms", [])
        if truth and row["decision"] == "reject":
            if findings:
                # Each truth interval gets its closest predicted interval; overbroad ranges remain penalized.
                errors.extend(min((abs(start-p[0]) + abs(end-p[1])) / 2 for p in findings) for start, end in truth)
                located_cases += 1
            else:
                unlocalized_rejects += 1
    usage_summary = {}
    for field in ("calls", "cost_usd", "latency_ms"):
        values = [r.get("usage", {}).get(field) for _, r in available]
        known = [value for value in values if value is not None]
        complete = len(known) == len(available) and not missing
        usage_summary[field] = {"known_sum": sum(known) if known else None,
                                "total": sum(known) if complete and known else None,
                                "unknown_judgments": len(values)-len(known),
                                "missing_judgments": len(missing), "complete": complete}
    return {"expected_cases": len(cases), "judged_cases": len(available), "missing_case_ids": missing,
            "coverage": _ratio(len(available), len(cases)),
            "severe_false_accept": _ratio(sum(r["decision"] == "accept" for _, r in severe), len(severe)),
            "severe_expected_cases": sum(c["ground_truth"]["severity"] == "severe" for c in cases),
            "false_reject": _ratio(sum(r["decision"] == "reject" for _, r in valid), len(valid)),
            "unknown": _ratio(unknown, len(available)),
            "unresolved_including_missing": _ratio(unknown+len(missing), len(cases)),
            "localization": {"mean_boundary_absolute_error_ms": statistics.mean(errors) if errors else None,
                             "localized_rejected_cases": located_cases, "localized_intervals": len(errors),
                             "unlocalized_rejected_cases": unlocalized_rejects},
            "usage": usage_summary}


def _comparability(document: dict) -> dict:
    configs = document.get("arm_configs", {})
    if not isinstance(configs, dict):
        raise EvaluationError("arm_configs must be an object")
    result = {}
    for other, factor in (("C", "local_review"), ("D", "fps")):
        baseline, candidate = configs.get("B"), configs.get(other)
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            result[f"B_vs_{other}"] = {"status": "unknown", "reason": "Missing declared arm configurations"}
            continue
        required = ("model", "prompt_sha256", "fps", "audio", "local_review", "other_parameters")
        if any(key not in item or item[key] is None for item in (baseline, candidate) for key in required):
            result[f"B_vs_{other}"] = {"status": "unknown", "reason": "Incomplete declared arm configurations"}
            continue
        expected = all(item[key] == ARMS[arm][key] for arm, item in (("B", baseline), (other, candidate))
                       for key in ("fps", "audio", "local_review"))
        differing = [key for key in set(baseline) | set(candidate) if key != factor and baseline.get(key) != candidate.get(key)]
        result[f"B_vs_{other}"] = {"status": "declared_comparable" if expected and not differing else "confounded",
                                   "changed_factor": factor, "unexpected_differing_fields": sorted(differing),
                                   "verified_execution": False}
    return result


def score(manifest_path: str | Path, judgments_path: str | Path) -> dict:
    """Score only supplied judgments; fixed tune/holdout results never get pooled."""
    manifest_path, judgments_path = Path(manifest_path).resolve(), Path(judgments_path).resolve()
    manifest, document = _load(manifest_path), _load(judgments_path)
    cases = _validate_manifest(manifest, manifest_path.parent)
    rows = _validate_judgments(document, cases)
    return {"schema_version": VERSION, "manifest_sha256": _file_digest(manifest_path),
            "judgments_sha256": _file_digest(judgments_path),
            "status": "supplied_judgments_scored", "model_calls_by_evaluator": 0,
            "comparability": _comparability(document),
            "results": {split: {arm: _metrics([c for c in cases.values() if c["split"] == split], rows, arm)
                                for arm in ARMS} for split in ("tune", "holdout")},
            "metric_notes": ["Error-rate denominators include supplied judgments only; missing coverage is reported separately.",
                             "Unknown and missing judgments are unresolved, never acceptance.",
                             "Localization is conditional on rejected defects with intervals; missed defects remain in false-accept/unknown metrics.",
                             "Unknown costs remain null; known_sum is only a partial sum, not a billing total.",
                             "tune and holdout are intentionally separate; no pooled acceptance claim is produced."],
            "limitations": LIMITATIONS}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", required=True)
    scorer = commands.add_parser("score")
    scorer.add_argument("--manifest", required=True)
    scorer.add_argument("--judgments", required=True)
    scorer.add_argument("--output", help="Optional new JSON result file")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.output) if args.command == "prepare" else score(args.manifest, args.judgments)
        if args.command == "score" and args.output:
            _write(Path(args.output).resolve(), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (EvaluationError, OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
