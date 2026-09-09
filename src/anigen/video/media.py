"""Local media evidence and technical QC. No model calls or semantic verdicts.

FFmpeg and ffprobe must be on PATH. Sparse frames are only visual evidence:
they cannot establish continuous motion, identity, dialogue, or audio quality.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


class MediaError(ValueError):
    """A local media operation cannot complete safely."""


LIMITATIONS = [
    "Technical QC does not assess character identity, story, continuity, or quality.",
    "Successful audio decoding does not establish intelligibility or audio/video sync.",
]


def _run(program: str, args: list[str], timeout: float = 300) -> subprocess.CompletedProcess:
    executable = shutil.which(program)
    if not executable:
        raise MediaError(f"Required program is missing from PATH: {program}")
    try:
        return subprocess.run(
            [executable, *args], capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError(f"{program} could not finish: {exc}") from exc


def _positive(value: object) -> float | None:
    try:
        number = float(str(value))
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _finite(value: object) -> float | None:
    try:
        number = float(str(value))
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(value: object) -> float | None:
    try:
        numerator, denominator = str(value).replace(":", "/").split("/")
        return _positive(float(numerator) / float(denominator))
    except (ValueError, ZeroDivisionError):
        return None


def _probe(path: Path, *, frames: bool = False, video_stream_index: int = 0) -> dict:
    args = ["-v", "error", "-protocol_whitelist", "file,pipe"]
    if frames:
        args += ["-select_streams", str(video_stream_index), "-show_frames", "-show_entries",
                 "frame=best_effort_timestamp,best_effort_timestamp_time,duration_time,pkt_duration_time,nb_samples"]
    else:
        args += ["-show_streams", "-show_format", "-show_data_hash", "sha256"]
    result = _run("ffprobe", [*args, "-of", "json", str(path)])
    if result.returncode or result.stderr.strip():
        raise MediaError(f"Cannot probe {path.name}: {result.stderr.strip()[-2000:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe did not return valid metadata for {path.name}") from exc


def _presentation_timing(source: Path, metadata: dict, video: dict) -> dict:
    """Keep video length distinct from the complete uploaded audiovisual timeline.

    A container duration may include a leading nonzero-PTS gap. Keep that raw
    value and reserve conservatively instead of assuming how a remote model
    normalizes it. Unknown stream timing cannot become a zero-second input.
    """
    streams = [stream for stream in metadata.get("streams", [])
               if stream.get("codec_type") in ("video", "audio")
               and not stream.get("disposition", {}).get("attached_pic")]
    timings, issues = [], []
    for stream in streams:
        start, duration = _finite(stream.get("start_time")), _positive(stream.get("duration"))
        timing_source = "stream_metadata"
        if start is None or duration is None:
            try:
                rows, pts = _frame_timeline(source, stream["index"])
                tail = (_positive(rows[-1].get("duration_time"))
                        or _positive(rows[-1].get("pkt_duration_time")))
                if tail is None and stream["codec_type"] == "audio":
                    samples, rate = _positive(rows[-1].get("nb_samples")), _positive(stream.get("sample_rate"))
                    if samples and rate:
                        tail = samples / rate
                if tail is not None:
                    start, duration = pts[0], pts[-1] + tail - pts[0]
                    timing_source = "decoded_frame_pts"
            except MediaError:
                pass
        timings.append({"stream_index": stream["index"], "type": stream["codec_type"],
                        "start_s": start, "duration_s": duration,
                        "end_s": start + duration if start is not None and duration is not None else None,
                        "timing_source": timing_source})
    complete = bool(timings) and all(row["end_s"] is not None for row in timings)
    start = min(row["start_s"] for row in timings) if complete else None
    end = max(row["end_s"] for row in timings) if complete else None
    span = end - start if complete else None
    container = _positive(metadata.get("format", {}).get("duration"))
    if not complete:
        issues.append("Some audio/video stream timing is unknown")
    main = next(row for row in timings if row["stream_index"] == video["index"])
    tolerance = 0.05  # Permits small codec priming/padding, not absent video seconds.
    if main["start_s"] is None or abs(main["start_s"]) > tolerance:
        issues.append("Primary video has a nonzero/unknown PTS origin; remote time normalization is unverified")
    for row in timings:
        if row["type"] == "audio" and row["end_s"] is not None and main["end_s"] is not None:
            if row["start_s"] < main["start_s"] - tolerance or row["end_s"] > main["end_s"] + tolerance:
                issues.append("Audio extends outside the primary video interval")
    if sum(row["type"] == "video" for row in timings) != 1 or sum(row["type"] == "audio" for row in timings) > 1:
        issues.append("AV observation requires one video and at most one audio stream")
    return {"container_duration_s": container, "presentation_streams": timings,
            "presentation_timing_complete": complete, "presentation_start_s": start,
            "presentation_end_s": end, "presentation_span_s": span,
            "media_seconds": max(span, container or 0) if complete else None,
            "video_start_relative_to_presentation_s": main["start_s"] - start if complete else None,
            "av_timeline_compatible": not issues, "av_timeline_tolerance_s": tolerance,
            "av_timeline_issues": issues}


def inspect_video(
    path: Path, expected_duration: float | None = None,
    expected_aspect: str | None = None, require_audio: bool = False,
) -> dict:
    """Probe the first video and decode video/audio fully; return an explicit verdict.

    Duration tolerance is max(0.15 seconds, two frames). Aspect tolerance is 1%.
    Invalid media returns ``ok=False``; unavailable programs/invalid expectations
    raise MediaError. No remote protocols are enabled for media inputs.
    """
    if expected_duration is not None and _positive(expected_duration) is None:
        raise MediaError("expected_duration must be finite and greater than zero")
    target_aspect = _ratio(expected_aspect) if expected_aspect is not None else None
    if expected_aspect is not None and target_aspect is None:
        raise MediaError("expected_aspect must be a positive ratio such as 16:9")
    # Check dependencies first so a missing decoder never looks like a failed video.
    for program in ("ffprobe", "ffmpeg"):
        if not shutil.which(program):
            raise MediaError(f"Required program is missing from PATH: {program}")
    source = Path(path).expanduser().resolve()
    report = {
        "ok": False, "path": str(source), "duration_s": None,
        "width": None, "height": None, "fps": None, "has_audio": False,
        "decode_ok": False, "decoded_frames": 0, "issues": [],
        "limitations": list(LIMITATIONS),
    }
    if not source.is_file():
        report["issues"].append("Input is not an existing regular file")
        return report
    try:
        metadata = _probe(source)
    except MediaError as exc:
        report["issues"].append(str(exc))
        return report
    videos = [s for s in metadata.get("streams", []) if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    audios = [s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"]
    report["has_audio"] = bool(audios)
    report["video_streams"] = len(videos)
    report["audio_streams"] = len(audios)
    if not videos:
        report["issues"].append("No playable video stream")
        return report
    video = videos[0]
    report["video_stream_index"] = video["index"]
    presentation = _presentation_timing(source, metadata, video)
    main_timing = next(row for row in presentation["presentation_streams"]
                       if row["stream_index"] == video["index"])
    video_start = main_timing["start_s"]
    report.update(video_start_pts=video.get("start_pts"), video_start_s=video_start,
                  video_time_base=video.get("time_base"),
                  format_start_s=_finite(metadata.get("format", {}).get("start_time")))
    # A container may last longer than its video (for example, a trailing audio
    # track). Missing stream duration must use decoded video timing, not that total.
    duration = main_timing["duration_s"]
    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate"))
    width, height = video.get("width", 0), video.get("height", 0)
    sar = _ratio(video.get("sample_aspect_ratio")) or 1.0
    rotation = next((s["rotation"] for s in video.get("side_data_list", [])
                     if "rotation" in s), 0)
    display_aspect = width * sar / height if width > 0 and height > 0 else None
    if display_aspect and abs(float(rotation)) % 180 == 90:
        display_aspect = 1 / display_aspect
    report.update(duration_s=duration, width=width, height=height, fps=fps,
                  display_aspect=display_aspect, rotation=rotation,
                  video_profile={key: video.get(key) for key in (
                      "codec_name", "profile", "level", "width", "height", "pix_fmt",
                      "sample_aspect_ratio", "r_frame_rate", "time_base", "extradata_hash",
                  )},
                  audio_profile=[{key: audio.get(key) for key in (
                      "codec_name", "sample_rate", "channels", "channel_layout",
                      "sample_fmt", "time_base", "extradata_hash",
                  )} for audio in audios])
    report["audio_timing"] = [{
        "stream_index": audio["index"], "start_pts": audio.get("start_pts"),
        "start_s": _finite(audio.get("start_time")), "time_base": audio.get("time_base"),
        "duration_s": _positive(audio.get("duration")),
        "sample_rate": audio.get("sample_rate"),
        "av_start_delta_s": (_finite(audio.get("start_time")) - video_start
                             if _finite(audio.get("start_time")) is not None and video_start is not None else None),
        "av_duration_delta_s": (_positive(audio.get("duration")) - duration
                                if _positive(audio.get("duration")) and duration else None),
    } for audio in audios]
    report["av_duration_delta_s"] = (report["audio_timing"][0]["av_duration_delta_s"]
                                      if len(audios) == 1 else None)
    if duration is None:
        report["issues"].append("No finite positive duration")
    if width <= 0 or height <= 0:
        report["issues"].append("Invalid video resolution")
    if fps is None:
        report["issues"].append("No finite positive frame rate")
    if require_audio and not audios:
        report["issues"].append("Required audio stream is absent")
    if duration and expected_duration is not None:
        tolerance = max(0.15, 2 / fps) if fps else 0.15
        if abs(duration - expected_duration) > tolerance:
            report["issues"].append(
                f"Duration {duration:.3f}s differs from expected {expected_duration:.3f}s "
                f"by more than {tolerance:.3f}s"
            )
    if target_aspect and display_aspect and abs(display_aspect / target_aspect - 1) > 0.01:
        report["issues"].append(f"Display aspect {display_aspect:.5f} differs from {expected_aspect}")
    try:
        decoded = _run("ffmpeg", [
            "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
            "-protocol_whitelist", "file,pipe", "-i", str(source),
            "-map", f"0:{video['index']}", "-map", "0:a?", "-progress", "pipe:1",
            "-nostats", "-fps_mode", "passthrough", "-f", "null", "-",
        ])
        frame_counts = [int(line.split("=", 1)[1]) for line in decoded.stdout.splitlines()
                        if line.startswith("frame=") and line.split("=", 1)[1].strip().isdigit()]
        report["decoded_frames"] = max(frame_counts, default=0)
        report["decode_ok"] = (decoded.returncode == 0 and not decoded.stderr.strip()
                               and report["decoded_frames"] > 0)
        if not report["decode_ok"]:
            report["issues"].append("Full decode failed or yielded no frames: "
                                    + decoded.stderr.strip()[-2000:])
    except MediaError as exc:
        report["issues"].append(str(exc))
    report.update(presentation)
    report["ok"] = not report["issues"]
    return report


def extract_frames(
    path: Path, output_dir: Path, interval_s: float = 1.0, *,
    boundary_times: list[float] | None = None,
) -> dict:
    """Extract actual decoded frames at start/end, intervals, and both sides of cuts.

    ``timestamp_s`` is the actual frame PTS relative to the first frame, never an
    invented seek time. ``boundary_times`` use that same relative timeline.
    At most 500 sampled frames are allowed. The output directory must be empty.
    """
    if _positive(interval_s) is None:
        raise MediaError("interval_s must be finite and greater than zero")
    source = Path(path).expanduser().resolve()
    inspection = inspect_video(source)
    if not inspection["ok"]:
        raise MediaError("Cannot extract frames: " + "; ".join(inspection["issues"]))
    if inspection["video_streams"] != 1:
        raise MediaError("Frame extraction requires exactly one video stream")
    rows = _probe(source, frames=True,
                  video_stream_index=inspection["video_stream_index"]).get("frames", [])
    try:
        timestamps = [float(row["best_effort_timestamp_time"]) for row in rows]
    except (KeyError, ValueError) as exc:
        raise MediaError("Video has frames without usable presentation timestamps") from exc
    if not timestamps or any(not math.isfinite(t) for t in timestamps):
        raise MediaError("No finite frame timestamps available")
    times = [time - timestamps[0] for time in timestamps]
    if any(right < left for left, right in zip(times, times[1:])):
        raise MediaError("Frame presentation timestamps are not monotonic")
    from bisect import bisect_left

    selected = {0, len(times) - 1}
    count = math.floor(times[-1] / interval_s)
    if count > 500:
        raise MediaError("Sampling would exceed 500 frames; increase interval_s")
    for step in range(1, count + 1):
        target = step * interval_s
        after = min(bisect_left(times, target), len(times) - 1)
        before = max(0, after - 1)
        selected.add(min((before, after), key=lambda index: abs(times[index] - target)))
    for boundary in boundary_times or []:
        if not isinstance(boundary, (int, float)) or not math.isfinite(boundary) or not 0 <= boundary <= inspection["duration_s"]:
            raise MediaError("Boundary times must be finite and within the video duration")
        after = min(bisect_left(times, boundary), len(times) - 1)
        selected.update((max(0, after - 1), after))
    indexes = sorted(selected)
    if len(indexes) > 500:
        raise MediaError("Sampling would exceed 500 frames; reduce boundaries or increase interval_s")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise MediaError("Frame output directory must be empty; existing evidence is never overwritten")
    expression = "+".join(f"eq(n\\,{index})" for index in indexes)
    result = _run("ffmpeg", [
        "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-i", str(source), "-map", f"0:{inspection['video_stream_index']}",
        "-vf", "select=" + expression, "-fps_mode", "vfr", "-n",
        str(destination / "frame_%06d.png"),
    ])
    if result.returncode or result.stderr.strip():
        raise MediaError("Frame extraction failed: " + result.stderr.strip()[-2000:])
    outputs = sorted(destination.glob("frame_*.png"))
    if len(outputs) != len(indexes):
        raise MediaError(f"Expected {len(indexes)} frames, produced {len(outputs)}")
    return {
        "path": str(source), "inspection": inspection,
        "frames": [{"timestamp_s": times[index], "source_pts_s": timestamps[index],
                    "frame_index": index, "path": str(output)}
                   for index, output in zip(indexes, outputs)],
        "limitations": [
            "Sparse frames do not verify continuous motion, intervening defects, or audio.",
            "Semantic acceptance requires an explicit reviewer verdict with supported evidence.",
        ],
    }


def _frame_timeline(path: Path, stream_index: int) -> tuple[list[dict], list[float]]:
    rows = _probe(path, frames=True, video_stream_index=stream_index).get("frames", [])
    points = [_finite(row.get("best_effort_timestamp_time")) for row in rows]
    if not points or any(point is None for point in points):
        raise MediaError("Every decoded frame must have a finite presentation timestamp")
    if any(right <= left for left, right in zip(points, points[1:])):
        raise MediaError("Window evidence requires strictly increasing frame timestamps")
    return rows, points


def inspect_window(path: Path, output_dir: Path, start_s: float, end_s: float,
                   fps: float = 8) -> dict:
    """Decode an exact source-time window and retain actual source-frame evidence.

    Times are relative to the source's first decoded video PTS. Only frames whose
    presentation begins in [start_s, end_s) are included; no preceding frame is
    invented to fill an off-grid beginning. The last packet's display duration is
    shortened at the requested end. Video timestamps use microsecond precision.
    Audio is re-encoded, retaining the source timeline: a leading stream gap is
    represented by digital silence, never by shifting the subsequent audio.
    Multi-track/rotated sources fail explicitly; the caller must normalize them.
    """
    if (isinstance(start_s, bool) or isinstance(end_s, bool) or isinstance(fps, bool)
            or _finite(start_s) is None or _finite(end_s) is None
            or not 0 <= start_s < end_s or _positive(fps) is None or fps > 120):
        raise MediaError("Window needs finite 0 <= start_s < end_s and 0 < fps <= 120")
    source = Path(path).expanduser().resolve()
    inspection = inspect_video(source)
    if not inspection["ok"]:
        raise MediaError("Cannot inspect window: " + "; ".join(inspection["issues"]))
    if inspection["video_streams"] != 1 or inspection["audio_streams"] > 1 or inspection["rotation"]:
        raise MediaError("Window extraction supports one unrotated video and at most one audio stream")
    if inspection["duration_s"] > 600:
        raise MediaError("Window inspection is limited to sources of at most 600 seconds")
    source_hash = _sha256(source)
    rows, pts = _frame_timeline(source, inspection["video_stream_index"])
    last_duration = _positive(rows[-1].get("duration_time")) or _positive(rows[-1].get("pkt_duration_time"))
    if last_duration is None:
        raise MediaError("Last decoded frame lacks a duration; exact source end is unknown")
    origin = pts[0]
    source_duration = pts[-1] + last_duration - origin
    if end_s > source_duration + 0.000001:
        raise MediaError("Window end lies outside the decoded video duration")
    absolute_start, absolute_end = origin + start_s, origin + end_s
    # Inputs finer than the output timestamp precision are rejected, not rounded silently.
    if any(abs(value * 1_000_000 - round(value * 1_000_000)) > 0.001
           for value in (absolute_start, absolute_end)):
        raise MediaError("Window boundaries must be representable to microsecond precision")
    indexes = [i for i, point in enumerate(pts)
               if absolute_start - 0.0000001 <= point < absolute_end - 0.0000001]
    if not indexes:
        raise MediaError("Window contains no actual video frame; it cannot be expanded implicitly")
    # Evidence sampling chooses actual frames, retaining the first/last frame in range.
    from bisect import bisect_left
    selected = {indexes[0], indexes[-1]}
    sample_count = math.ceil((end_s - start_s) * fps)
    if sample_count > 500:
        raise MediaError("Window sampling would exceed 500 frames")
    for step in range(sample_count):
        after = bisect_left(pts, absolute_start + step / fps)
        candidates = [i for i in (after - 1, after) if indexes[0] <= i <= indexes[-1]]
        if candidates:
            selected.add(min(candidates, key=lambda i: abs(pts[i] - absolute_start - step / fps)))
    if len(selected) > 500:
        raise MediaError("Window sampling would exceed 500 frames")
    audio_overlap = False
    if inspection["has_audio"]:
        audio = inspection["audio_timing"][0]
        audio_rows, audio_pts = _frame_timeline(source, audio["stream_index"])
        audio_tail = (_positive(audio_rows[-1].get("duration_time"))
                      or _positive(audio_rows[-1].get("pkt_duration_time")))
        if audio_tail is None:
            raise MediaError("Audio end is unknown; cannot preserve an exact window")
        audio_overlap = audio_pts[0] < absolute_end and audio_pts[-1] + audio_tail > absolute_start
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise MediaError("Window output directory must be empty; existing evidence is never overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Build off to the side and publish only after hashes and PTS are validated.
    with tempfile.TemporaryDirectory(prefix=".window-", dir=destination.parent) as temporary:
        temporary_dir = Path(temporary)
        pending = temporary_dir / "window.mp4"
        filters = [f"[0:{inspection['video_stream_index']}]trim=start={absolute_start:.6f}:end={absolute_end:.6f},"
                   f"settb=1/1000000,setpts=PTS-{absolute_start:.6f}*1000000[v]"]
        maps = ["-map", "[v]"]
        if audio_overlap:
            filters.append(f"[0:{audio['stream_index']}]atrim=start={absolute_start:.6f}:end={absolute_end:.6f},"
                           f"asetpts=PTS-{absolute_start:.6f}/TB,aresample=async=1:first_pts=0[a]")
            maps += ["-map", "[a]"]
        tail_duration = min(absolute_end - pts[indexes[-1]],
                            _positive(rows[indexes[-1]].get("duration_time"))
                            or _positive(rows[indexes[-1]].get("pkt_duration_time"))
                            or (pts[indexes[-1] + 1] - pts[indexes[-1]] if indexes[-1] + 1 < len(pts) else last_duration))
        command = ["-nostdin", "-v", "error", "-xerror", "-protocol_whitelist", "file,pipe",
                   "-copyts", "-i", str(source), "-filter_complex", ";".join(filters), *maps,
                   "-c:v", "libx264", "-crf", "18", "-bf", "0", "-enc_time_base:v", "1/1000000",
                   "-video_track_timescale", "1000000", "-fps_mode", "passthrough",
                   "-bsf:v", f"setts=duration=if(eq(N\\,{len(indexes)-1})\\,{tail_duration:.6f}/TB\\,DURATION)"]
        if audio_overlap:
            command += ["-c:a", "aac", "-b:a", "192k"]
        command += ["-avoid_negative_ts", "disabled", "-n", str(pending)]
        result = _run("ffmpeg", command, timeout=180)
        if result.returncode or result.stderr.strip():
            raise MediaError("Precise window decode/encode failed: " + result.stderr.strip()[-2000:])
        crop_inspection = inspect_video(pending, require_audio=audio_overlap)
        if not crop_inspection["ok"]:
            raise MediaError("Window failed technical QC: " + "; ".join(crop_inspection["issues"]))
        crop_rows, crop_pts = _frame_timeline(pending, crop_inspection["video_stream_index"])
        if (len(crop_pts) != len(indexes)
                or any(abs(actual - (pts[index] - absolute_start)) > 0.000002
                       for index, actual in zip(indexes, crop_pts))):
            raise MediaError("Encoded window timestamps differ from source frame mapping")
        crop_tail = _positive(crop_rows[-1].get("duration_time")) or _positive(crop_rows[-1].get("pkt_duration_time"))
        if crop_tail is None or crop_pts[-1] + crop_tail > end_s - start_s + 0.000002:
            raise MediaError("Encoded window extends beyond the requested end")
        if audio_overlap and crop_inspection["audio_streams"] != 1:
            raise MediaError("Window lost the source audio stream")
        chosen = sorted(selected)
        frame_dir = temporary_dir / "frames"
        frame_dir.mkdir()
        expression = "+".join(f"eq(n\\,{index})" for index in chosen)
        result = _run("ffmpeg", ["-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
                                 "-i", str(source), "-map", f"0:{inspection['video_stream_index']}",
                                 "-vf", "select=" + expression, "-fps_mode", "vfr", "-n",
                                 str(frame_dir / "frame_%06d.png")], timeout=180)
        outputs = sorted(frame_dir.glob("frame_*.png"))
        if result.returncode or result.stderr.strip() or len(outputs) != len(chosen):
            raise MediaError("Window source frame extraction failed: " + result.stderr.strip()[-2000:])
        if _sha256(source) != source_hash:
            raise MediaError("Source changed during window inspection")
        destination.mkdir(exist_ok=True)
        if any(destination.iterdir()):
            raise MediaError("Window output appeared during extraction; refusing overwrite")
        published_frames = destination / "frames"
        published_frames.mkdir()
        # Each publication is an atomic create, with no replace semantics.
        os.link(pending, destination / pending.name)
        frames = []
        for index, output in zip(chosen, outputs):
            final = published_frames / output.name
            os.link(output, final)
            frames.append({"path": str(final), "sha256": _sha256(output),
                           "source_frame_index": index, "source_pts_s": pts[index],
                           "source_timestamp_s": pts[index] - origin,
                           "crop_timestamp_s": pts[index] - absolute_start})
        crop_inspection["path"] = str(destination / pending.name)
        return {
            "source_path": str(source), "source_sha256": source_hash,
            "source_duration_s": source_duration, "source_pts_start_s": origin,
            "source_time_base": inspection["video_time_base"],
            "requested_window_s": {"start": start_s, "end": end_s}, "requested_fps": fps,
            "crop": {"path": str(destination / pending.name), "sha256": _sha256(pending),
                     "timeline_duration_s": end_s - start_s, "inspection": crop_inspection},
            "crop_to_source_time_mapping": {
                "source_time_origin_s": start_s, "source_absolute_pts_origin_s": absolute_start,
                "method": "source_timestamp_s = crop_timestamp_s + source_time_origin_s",
                "timestamp_precision_s": 0.000001,
                "provider_timestamp_origin": "unverified; mapping assumes MP4 container timeline, not first video frame",
            },
            "audio_stream_present_in_source": inspection["has_audio"],
            "audio_overlap_in_window": audio_overlap,
            "audio_processing": "AAC re-encode; leading gaps become digital silence without moving events"
                                if audio_overlap else "No source audio samples overlap this window",
            "frame_mapping": [{"source_frame_index": index, "source_pts_s": pts[index],
                               "source_timestamp_s": pts[index] - origin, "crop_timestamp_s": actual}
                              for index, actual in zip(indexes, crop_pts)],
            "frames": frames,
            "limitations": [
                "Crop is re-encoded; dense images are extracted from the original source.",
                "An off-grid window may begin before its first actual video frame; no image is invented.",
                "Source PTS are measured; model event timestamps remain estimates.",
                "Provider timestamp origin must be contract-tested: normalizing to the first video frame would shift off-grid window times.",
                "Audio is re-encoded and leading stream gaps are represented as silence; this is not bit-exact audio.",
            ],
        }


def signal_stats(path: Path) -> dict:
    """Return bounded deterministic anomaly candidates, never semantic failures."""
    source = Path(path).expanduser().resolve()
    inspection = inspect_video(source)
    if not inspection["ok"]:
        raise MediaError("Cannot measure signals: " + "; ".join(inspection["issues"]))
    if inspection["duration_s"] > 600:
        raise MediaError("Signal statistics are limited to sources of at most 600 seconds")
    source_hash = _sha256(source)
    _, pts = _frame_timeline(source, inspection["video_stream_index"])
    origin = pts[0]
    thresholds = {"black_pixel_threshold": 0.1, "black_picture_ratio": 0.98,
                  "black_min_duration_s": 0.05, "freeze_noise_db": -60,
                  "freeze_min_duration_s": 0.5, "silence_noise_db": -50,
                  "silence_min_duration_s": 0.2, "near_full_scale_dbfs": -0.1}
    result = _run("ffmpeg", ["-nostdin", "-v", "info", "-protocol_whitelist", "file,pipe",
                             "-copyts", "-i", str(source), "-map", f"0:{inspection['video_stream_index']}",
                             "-vf", "blackdetect=d=0.05:pix_th=0.1:pic_th=0.98,freezedetect=n=-60dB:d=0.5",
                             "-an", "-f", "null", "-"], timeout=180)
    if result.returncode:
        raise MediaError("Visual signal measurement failed: " + result.stderr[-2000:])
    def intervals(log: str, kind: str) -> list[dict]:
        found, pending = [], None
        pattern = rf"\b{kind}_(start|end):\s*(-?[0-9.]+)"
        for match in re.finditer(pattern, log):
            time = float(match[2]) - origin
            if match[1] == "start":
                pending = time
            elif pending is not None:
                found.append({"start_s": pending, "end_s": time, "end_censored": False})
                pending = None
        if pending is not None:
            found.append({"start_s": pending, "end_s": inspection["duration_s"], "end_censored": True})
        return found
    report = {"source_path": str(source), "source_sha256": source_hash,
              "source_pts_start_s": origin, "thresholds": thresholds,
              "black_intervals": intervals(result.stderr, "black"),
              "freeze_intervals": intervals(result.stderr, "freeze"), "audio": [],
              "limitations": [
                  "All statistics are warning candidates, not acceptance or rejection decisions.",
                  "Intentional animation holds, black transitions and silence may be correct.",
                  "A near-full-scale peak is only a clipping candidate, not proof of audible distortion.",
                  "Intervals use the source first-video-frame timeline; audio may start before zero.",
              ]}
    for audio in inspection["audio_timing"]:
        result = _run("ffmpeg", ["-nostdin", "-v", "info", "-protocol_whitelist", "file,pipe",
                                 "-copyts", "-i", str(source), "-map", f"0:{audio['stream_index']}",
                                 "-af", "silencedetect=n=-50dB:d=0.2,astats=metadata=0:reset=0",
                                 "-vn", "-f", "null", "-"], timeout=180)
        if result.returncode:
            raise MediaError("Audio signal measurement failed: " + result.stderr[-2000:])
        peaks = re.findall(r"Peak level dB:\s*(-?(?:[0-9.]+|inf))", result.stderr)
        finite_peaks = [_finite(value) for value in peaks if _finite(value) is not None]
        peak = max(finite_peaks) if finite_peaks else None
        report["audio"].append({"stream_index": audio["stream_index"],
                                "silence_intervals": intervals(result.stderr, "silence"),
                                "peak_dbfs": peak, "digital_silence": bool(peaks) and all(value == "-inf" for value in peaks),
                                "clipping_candidate": peak is not None and peak >= -0.1})
    if _sha256(source) != source_hash:
        raise MediaError("Source changed during signal measurement")
    return report


def inspect_audio(path: Path) -> dict:
    """Measure the actual uploaded audio timeline; never trust declared duration."""
    source = Path(path).expanduser().resolve()
    metadata = _probe(source)
    streams = metadata.get("streams", [])
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if len(audios) != 1 or any(s.get("codec_type") == "video" for s in streams):
        raise MediaError("Audio reference must have one audio stream and no video")
    durations = [_positive(audios[0].get("duration")), _positive(metadata.get("format", {}).get("duration"))]
    duration = max((d for d in durations if d is not None), default=None)
    if duration is None:
        raise MediaError("Audio reference has no finite duration")
    result = _run("ffmpeg", ["-nostdin", "-v", "error", "-xerror", "-protocol_whitelist", "file,pipe",
                             "-i", str(source), "-map", "0:a:0", "-f", "null", "-"])
    if result.returncode or result.stderr.strip():
        raise MediaError("Audio reference failed full decoding")
    return {"path": str(source), "duration_s": duration, "sha256": _sha256(source)}


def assemble(paths: list[Path], output: Path) -> dict:
    """Concatenate compatible clips in caller order, preserve audio, and decode QC.

    Stream copying avoids generation loss. Incompatible stream profiles, mixed
    audio, and multiple video/audio tracks are rejected instead of silently
    normalizing or dropping content. Existing output files are never overwritten.
    """
    if not paths:
        raise MediaError("At least one clip is required")
    sources = [Path(path).expanduser().resolve() for path in paths]
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination in sources:
        raise MediaError("Assembly output must be a new file")
    if destination.suffix.lower() not in (".mp4", ".mov", ".mkv"):
        raise MediaError("Assembly output extension must be .mp4, .mov, or .mkv")
    inspections = [inspect_video(source) for source in sources]
    for item in inspections:
        if not item["ok"]:
            raise MediaError(f"Clip failed QC: {item['path']}: {'; '.join(item['issues'])}")
        if item["video_streams"] != 1 or item["audio_streams"] > 1:
            raise MediaError("Assembly requires one video and at most one audio track per clip")
    first = inspections[0]
    for item in inspections[1:]:
        if item["has_audio"] != first["has_audio"]:
            raise MediaError("Mixed audio presence: every clip must either have audio or have no audio")
        if item["video_profile"] != first["video_profile"] or item["rotation"] != first["rotation"]:
            raise MediaError("Incompatible video stream profiles; normalize clips explicitly before assembly")
        if item["audio_profile"] != first["audio_profile"]:
            raise MediaError("Incompatible audio stream profiles; normalize clips explicitly before assembly")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Private temporary manifest: quote paths for FFmpeg's concat format, not a shell.
    with tempfile.TemporaryDirectory(prefix=".assembly-", dir=destination.parent) as temporary:
        temporary_dir = Path(temporary)
        manifest = temporary_dir / "clips.ffconcat"
        lines = ["ffconcat version 1.0"]
        for source in sources:
            if "\n" in str(source) or "\r" in str(source):
                raise MediaError("Newlines in clip paths are not supported by concat manifests")
            escaped = str(source).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pending = temporary_dir / ("assembled" + destination.suffix.lower())
        result = _run("ffmpeg", [
            "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-f", "concat", "-safe", "0", "-i", str(manifest),
            "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-n", str(pending),
        ])
        if result.returncode or result.stderr.strip():
            raise MediaError("Assembly failed: " + result.stderr.strip()[-2000:])
        expected = sum(item["duration_s"] for item in inspections)
        inspection = inspect_video(pending, expected_duration=expected,
                                   expected_aspect=f"{first['display_aspect']}:1",
                                   require_audio=first["has_audio"])
        if not inspection["ok"]:
            raise MediaError("Assembled video failed QC: " + "; ".join(inspection["issues"]))
        if inspection["decoded_frames"] != sum(item["decoded_frames"] for item in inspections):
            raise MediaError("Assembled video frame count differs from the input total")
        # Atomic create without replacing a file that appeared while assembling.
        try:
            os.link(pending, destination)
        except OSError as exc:
            raise MediaError(f"Could not publish assembly to {destination}: {exc}") from exc
    inspection["path"] = str(destination)
    return {"path": str(destination), "clips": [str(path) for path in sources],
            "inspection": inspection}


def visual_only_copy(source: Path, directory: Path) -> Path:
    """Stream-copy video without audio for visual review; retain original output."""
    source = Path(source)
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(source.read_bytes()).hexdigest()
    output = directory / (name + '.mp4')
    # Recreate deterministically from the bound source; do not trust a stale copy.
    result = _run('ffmpeg', ['-nostdin', '-v', 'error', '-y', '-i', str(source),
                            '-map', '0:v:0', '-c:v', 'copy', '-an', str(output)])
    if result.returncode:
        raise MediaError('Cannot create visual-only observation copy')
    original, derived = inspect_video(source), inspect_video(output)
    if not derived['ok'] or derived['has_audio'] or abs(original['duration_s'] - derived['duration_s']) > 0.05:
        raise MediaError('Visual-only copy failed duration/audio verification')
    return output
