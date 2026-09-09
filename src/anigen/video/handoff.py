"""Local, immutable end-of-segment packages; no uploads or model requests."""
from __future__ import annotations

import math
from pathlib import Path
import shutil

from . import media
from .observation import file_hash


STATE_FIELDS = ("camera", "character_positions", "motion", "screen_direction", "environment", "next_action")


def export(observation, output, continuity_state, tail_duration_s=1):
    """Retain measured source PTS, the actual last decoded frame, and a tail clip.

    A short tail is useful locally. The current generation API accepts a video
    reference only when it is at least two seconds; callers enforce that limit.
    """
    if (type(tail_duration_s) not in (int, float) or not math.isfinite(tail_duration_s)
            or not 0.5 <= tail_duration_s <= 2):
        raise ValueError("handoff tail_duration_s must be finite in 0.5..2 seconds")
    if not isinstance(continuity_state, dict) or not all(
            isinstance(continuity_state.get(key), str) and continuity_state[key].strip() for key in STATE_FIELDS):
        raise ValueError("handoff continuity_state needs camera, character_positions, motion, screen_direction, environment, next_action")
    source = Path(observation["media_path"])
    if file_hash(source) != observation["media_sha256"]:
        raise ValueError("handoff source media changed since observation")
    # Collected observations always sample the last actual source frame. Match
    # it to the window output rather than substituting duration-minus-epsilon.
    final_source = observation["frames"][-1]
    duration = observation["qc"]["duration_s"]
    output = Path(output)
    window = media.inspect_window(source, output / "ending", duration - tail_duration_s, duration, fps=8)
    last = window["frames"][-1]
    if (last["source_frame_index"] != final_source["frame_index"]
            or abs(last["source_pts_s"] - final_source["source_pts_s"]) > 0.000002
            or last["sha256"] != final_source["sha256"]):
        raise ValueError("handoff ending does not contain the accepted source's actual final frame")
    final_path = output / "final_frame.png"
    with final_path.open("xb") as target, Path(last["path"]).open("rb") as stream:
        shutil.copyfileobj(stream, target)
    artifacts = {
        "final_frame": {"name": final_path.name, "path": str(final_path), "sha256": file_hash(final_path),
                        "mime_type": "image/png", "source_frame_index": last["source_frame_index"],
                        "source_timestamp_s": last["source_timestamp_s"]},
        "tail_video": {"name": "window.mp4", "path": window["crop"]["path"],
                       "sha256": window["crop"]["sha256"], "mime_type": "video/mp4",
                       "duration_s": window["crop"]["inspection"]["duration_s"],
                       "source_start_s": window["requested_window_s"]["start"],
                       "source_end_s": window["requested_window_s"]["end"],
                       "contains_final_frame": True},
    }
    return {"continuity_state": continuity_state, "tail_duration_s": tail_duration_s,
            "artifacts": artifacts, "ending": window,
            "limitations": ["Director-provided continuity notes are not automatically inferred or verified.",
                            "A local package is not uploaded. Bind an actual uploaded artifact URL before preparing the next segment.",
                            "The tail clip is re-encoded; final_frame is an original decoded source frame."]}
