"""Bounded local technical derivation; no generated frames, audio fill or network."""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path

from . import media


def _require(condition, message):
    if not condition:
        raise media.MediaError(message)


def normalize(source, destination, *, duration_s, aspect_ratio):
    """Fit and pad the full image, retain timing, and remove at most 0.2 s at tail.

    CFR sources whose planned duration is an exact number of frames are supported.
    Unsupported timing is rejected instead of resampling or duplicating frames.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    _require(source.is_file() and not destination.exists() and source != destination,
             "normalization needs an existing source and a new output file")
    _require(type(duration_s) in (int, float) and math.isfinite(duration_s) and duration_s > 0,
             "normalization duration must be finite and positive")
    try:
        ratio = Fraction(aspect_ratio.replace(":", "/"))
    except (AttributeError, ValueError, ZeroDivisionError):
        raise media.MediaError("normalization needs the planned numeric aspect ratio")
    _require(ratio > 0, "normalization aspect must be positive")
    source_hash = media._sha256(source)
    metadata = media._probe(source)
    source_qc = media.inspect_video(source)
    _require(source_qc["ok"] and source_qc["decode_ok"], "normalization cannot repair failed decoding")
    _require(source_qc["video_streams"] == 1 and source_qc["audio_streams"] <= 1,
             "normalization requires one video and at most one audio stream")
    video = next(s for s in metadata["streams"] if s["index"] == source_qc["video_stream_index"])
    _require(not any(float(s.get("rotation", 0)) % 360 for s in video.get("side_data_list", [])),
             "rotated media needs an explicit supported normalization path")
    source_duration = source_qc["duration_s"]
    trim_s = source_duration - duration_s
    _require(trim_s >= -0.000001, "normalization must not extend or duplicate a short video")
    _require(trim_s <= 0.2 + 0.000001, "normalization permits at most 0.2 seconds of tail removal")
    fps = Fraction(video.get("avg_frame_rate", "0/1"))
    frame_count = fps * Fraction(str(duration_s))
    _require(fps > 0 and frame_count.denominator == 1,
             "exact duration requires a whole source frame count; no frame-rate resampling is allowed")
    _, pts = media._frame_timeline(source, video["index"])
    origin = pts[0]
    _require(all(abs((b - a) - float(1 / fps)) <= 0.001 for a, b in zip(pts, pts[1:])),
             "variable frame timing is not supported without explicit review")
    _require(len(pts) >= int(frame_count), "normalization cannot synthesize missing source frames")
    _require(abs(pts[int(frame_count) - 1] - origin - (int(frame_count) - 1) / float(fps)) <= 0.001,
             "source frames do not cover the exact target timeline")
    for timing in source_qc["presentation_streams"]:
        if timing["type"] == "audio":
            _require(timing["start_s"] is not None and timing["end_s"] is not None
                     and abs(timing["start_s"] - origin) <= 0.05
                     and timing["end_s"] >= origin + duration_s - 0.05
                     and timing["end_s"] <= origin + source_duration + 0.05,
                     "audio timing outside the video cannot be repaired by filling or shifting audio")

    sar = media._ratio(video.get("sample_aspect_ratio")) or 1.0
    display_width, height = video["width"] * sar, video["height"]
    # Even dimensions, exact planned ratio, and no upscale of the source viewport.
    factor = math.floor(min(display_width / ratio.numerator, height / ratio.denominator) / 2) * 2
    width, out_height = factor * ratio.numerator, factor * ratio.denominator
    _require(factor >= 2 and max(width, out_height) <= 8192, "unsupported output geometry")
    scale = min(width / display_width, out_height / height)
    fit_width = max(2, math.floor(display_width * scale / 2) * 2)
    fit_height = max(2, math.floor(height * scale / 2) * 2)
    vf = (f"trim=start={origin:.9f}:end={origin + duration_s:.9f},setpts=PTS-{origin:.9f}/TB,"
          f"scale={fit_width}:{fit_height},setsar=1,pad={width}:{out_height}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p")
    args = ["-nostdin", "-v", "error", "-n", "-copyts", "-noautorotate", "-protocol_whitelist", "file,pipe",
            "-i", str(source), "-map", f"0:{video['index']}", "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-fps_mode", "passthrough",
            "-video_track_timescale", "90000"]
    if source_qc["has_audio"]:
        args += ["-map", "0:a:0", "-af", f"atrim=start={origin:.9f}:end={origin + duration_s:.9f},asetpts=PTS-{origin:.9f}/TB",
                 "-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-an"]
    args += ["-map_metadata", "-1", "-t", str(duration_s), "-movflags", "+faststart", str(destination)]
    result = media._run("ffmpeg", args)
    _require(result.returncode == 0 and not result.stderr.strip(), "technical normalization failed: " + result.stderr[-1500:])
    _require(media._sha256(source) == source_hash, "normalization source changed during processing")
    output_metadata = media._probe(destination)
    output_qc = media.inspect_video(destination, expected_duration=duration_s, expected_aspect=aspect_ratio,
                                    require_audio=source_qc["has_audio"])
    _require(output_qc["ok"] and output_qc["av_timeline_compatible"], "normalized output failed technical QC")
    _require(abs(output_qc["duration_s"] - duration_s) <= 0.00001
             and abs(output_qc["container_duration_s"] - duration_s) <= 0.001,
             "normalized output did not reach the exact planned duration")
    _require(output_qc["decoded_frames"] == int(frame_count), "normalization changed the source frame count")
    _require(output_qc["has_audio"] == source_qc["has_audio"], "normalization changed audio presence")
    return {"operation": "fit_pad_tail_trim", "source_path": str(source), "source_sha256": source_hash,
            "output_path": str(destination), "output_sha256": media._sha256(destination),
            "source_metadata": metadata, "source_decode_qc": source_qc,
            "output_metadata": output_metadata, "output_qc": output_qc,
            "planned_duration_s": duration_s, "tail_removed_s": max(0, trim_s), "max_tail_removed_s": 0.2,
            "source_origin_s": origin, "output_origin_s": output_qc["video_start_s"],
            "aspect_ratio": aspect_ratio, "canvas": {"width": width, "height": out_height},
            "fit": {"width": fit_width, "height": fit_height}, "full_frame_retained": True,
            "frame_rate_resampled": False, "frames_added": 0, "output_frame_count": int(frame_count),
            "audio_presence_preserved": True, "audio_fill_added": False,
            "limitations": ["Technical derivation only; semantic, action and continuity review are still required.",
                            "Audio is reencoded without fill; exact synchronization still requires observation."]}
