"""Offline observation fixtures exercise real decoder timestamps and audio timing."""

from array import array
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from anigen.video.media import MediaError, inspect_video, inspect_window, signal_stats


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
class MediaObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="media-observation-test-")
        cls.root = Path(cls.temporary.name)
        cls.offset = cls.clip("offset.mp4", offset=0.5)
        cls.nonzero = cls.clip("nonzero.mp4", offset=0.5, pts_offset=5)
        cls.noaudio = cls.clip("noaudio.mp4", audio=False)
        cls.black = cls.clip("black.mp4", visual="color=c=black:s=160x90:r=10:d=2", quiet=True)
        cls.peak = cls.clip("peak.mp4", constant_peak=True)
        cls.longer = cls.clip("longer.mp4", audio=False, visual="testsrc2=s=160x90:r=10:d=6")
        cls.long_audio = cls.clip("long-audio.mp4", audio_duration=15,
                                  visual="testsrc2=s=160x90:r=10:d=5")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def clip(cls, name, *, offset=0, pts_offset=0, audio=True, quiet=False,
             constant_peak=False, visual="testsrc2=s=160x90:r=10:d=2", audio_duration=1.4):
        path = cls.root / name
        args = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", visual]
        if audio:
            signal = "0" if quiet else "0.9999" if constant_peak else "if(between(t,0.2,0.22),0.8,0)"
            args += ["-itsoffset", str(offset), "-f", "lavfi", "-i",
                     f"aevalsrc='{signal}':s=48000:d={audio_duration}", "-c:a", "alac"]
        args += ["-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p",
                 "-output_ts_offset", str(pts_offset), str(path)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return path

    def test_stream_timing_is_explicit(self):
        report = inspect_video(self.nonzero)
        self.assertTrue(report["ok"])
        self.assertAlmostEqual(report["video_start_s"], 5)
        self.assertIsInstance(report["video_start_pts"], int)
        self.assertTrue(report["video_time_base"])
        self.assertAlmostEqual(report["audio_timing"][0]["av_start_delta_s"], 0.5)
        self.assertAlmostEqual(report["av_duration_delta_s"], -0.6, places=5)
        self.assertAlmostEqual(report["presentation_start_s"], 5)
        self.assertAlmostEqual(report["presentation_span_s"], 2)
        self.assertGreaterEqual(report["media_seconds"], report["presentation_span_s"])
        self.assertFalse(report["av_timeline_compatible"])

    def test_long_audio_cannot_hide_inside_short_video_budget(self):
        report = inspect_video(self.long_audio, expected_duration=5, require_audio=True)
        self.assertTrue(report["ok"], report)
        self.assertAlmostEqual(report["duration_s"], 5)
        self.assertAlmostEqual(report["container_duration_s"], 15)
        self.assertAlmostEqual(report["presentation_span_s"], 15)
        self.assertAlmostEqual(report["media_seconds"], 15)
        self.assertTrue(report["presentation_timing_complete"])
        self.assertFalse(report["av_timeline_compatible"])
        self.assertIn("Audio extends outside the primary video interval", report["av_timeline_issues"])
        audio = next(row for row in report["presentation_streams"] if row["type"] == "audio")
        self.assertAlmostEqual(audio["end_s"], 15)
        self.assertTrue(inspect_video(self.offset)["av_timeline_compatible"])
        # Matroska omits per-stream duration here; container duration must not
        # replace the shorter video duration when decoded PTS can establish it.
        remuxed = self.root / "long-audio.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.long_audio), "-map", "0",
                        "-c", "copy", str(remuxed)], check=True, timeout=30)
        fallback = inspect_video(remuxed, expected_duration=5)
        self.assertTrue(fallback["ok"], fallback)
        self.assertAlmostEqual(fallback["duration_s"], 5)
        self.assertGreaterEqual(fallback["media_seconds"], 15)
        self.assertFalse(fallback["av_timeline_compatible"])

    def test_off_grid_window_retains_pts_and_truncates_last_frame(self):
        report = inspect_window(self.offset, self.root / "offgrid", 0.26, 1.27)
        self.assertAlmostEqual(report["frame_mapping"][0]["source_timestamp_s"], 0.3)
        self.assertAlmostEqual(report["frame_mapping"][0]["crop_timestamp_s"], 0.04)
        self.assertAlmostEqual(report["crop"]["inspection"]["video_start_s"], 0.04)
        self.assertAlmostEqual(report["crop"]["inspection"]["duration_s"], 0.97)
        self.assertAlmostEqual(report["crop"]["timeline_duration_s"], 1.01)
        self.assertTrue(report["crop"]["inspection"]["has_audio"])
        for frame in report["frames"]:
            self.assertEqual(hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest(), frame["sha256"])
            self.assertGreaterEqual(frame["source_timestamp_s"], 0.26)
            self.assertLess(frame["source_timestamp_s"], 1.27)
            self.assertAlmostEqual(frame["source_timestamp_s"], frame["crop_timestamp_s"] + 0.26)
        self.assertEqual(report["source_sha256"], hashlib.sha256(self.offset.read_bytes()).hexdigest())
        self.assertEqual(report["crop"]["sha256"], hashlib.sha256(Path(report["crop"]["path"]).read_bytes()).hexdigest())

    def test_audio_event_offset_is_preserved_for_nonzero_source_pts(self):
        report = inspect_window(self.nonzero, self.root / "nonzero-window", 0.26, 1.27, fps=20)
        self.assertAlmostEqual(report["source_pts_start_s"], 5)
        self.assertAlmostEqual(report["frame_mapping"][0]["source_pts_s"], 5.3)
        self.assertAlmostEqual(report["frame_mapping"][0]["crop_timestamp_s"], 0.04)
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", report["crop"]["path"],
                                 "-map", "0:a:0", "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                                capture_output=True, timeout=30, check=True)
        samples = array("f", result.stdout)
        first_loud = next(index for index, value in enumerate(samples) if abs(value) > 0.1) / 48000
        # Source audio begins at source-relative .5, impulse begins .2 later.
        self.assertAlmostEqual(first_loud, 0.5 + 0.2 - 0.26, delta=0.002)
        self.assertEqual(len(report["frames"]), len(report["frame_mapping"]))

    def test_short_window_does_not_expand_or_invent_audio(self):
        report = inspect_window(self.offset, self.root / "short-window", 0.26, 0.32)
        self.assertEqual(len(report["frame_mapping"]), 1)
        self.assertAlmostEqual(report["crop"]["inspection"]["duration_s"], 0.02)
        self.assertFalse(report["crop"]["inspection"]["has_audio"])
        self.assertTrue(report["audio_stream_present_in_source"])
        self.assertFalse(report["audio_overlap_in_window"])
        with self.assertRaisesRegex(MediaError, "no actual video frame"):
            inspect_window(self.offset, self.root / "no-frame", 0.21, 0.25)
        self.assertFalse((self.root / "no-frame").exists())

    def test_no_audio_and_full_end_boundary(self):
        report = inspect_window(self.noaudio, self.root / "noaudio-window", 0, 2)
        self.assertFalse(report["audio_stream_present_in_source"])
        self.assertFalse(report["crop"]["inspection"]["has_audio"])
        self.assertEqual(len(report["frame_mapping"]), 20)
        self.assertAlmostEqual(report["source_duration_s"], 2)

    def test_invalid_limits_and_no_overwrite(self):
        for start, end, fps in [(-1, 1, 8), (1, 1, 8), (0, 2.1, 8),
                                (0, 1, 0), (0, 1, float("nan")), (True, 2, 8),
                                (0.1234567, 1, 8)]:
            with self.assertRaises(MediaError):
                inspect_window(self.noaudio, self.root / "invalid", start, end, fps)
        output = self.root / "existing-window"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("existing")
        with self.assertRaisesRegex(MediaError, "empty"):
            inspect_window(self.noaudio, output, 0, 1)
        self.assertEqual(marker.read_text(), "existing")
        with self.assertRaisesRegex(MediaError, "500"):
            inspect_window(self.longer, self.root / "too-many-frames", 0, 6, 100)

    def test_signal_candidates_do_not_become_semantic_verdicts(self):
        report = signal_stats(self.black)
        self.assertTrue(report["black_intervals"])
        self.assertTrue(report["freeze_intervals"])
        self.assertTrue(report["audio"][0]["silence_intervals"])
        self.assertTrue(report["audio"][0]["digital_silence"])
        self.assertFalse(report["audio"][0]["clipping_candidate"])
        self.assertNotIn("ok", report)
        self.assertNotIn("decision", report)
        self.assertEqual(report["thresholds"]["freeze_min_duration_s"], 0.5)
        self.assertTrue(signal_stats(self.peak)["audio"][0]["clipping_candidate"])
        self.assertEqual(signal_stats(self.noaudio)["audio"], [])


if __name__ == "__main__":
    unittest.main()
