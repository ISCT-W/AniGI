"""Offline synthetic-media checks; all inputs and outputs live in a temp directory."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from anigen.video.media import MediaError, assemble, extract_frames, inspect_video


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe unavailable")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="gpt-agent-media-test-")
        cls.root = Path(cls.temporary.name).resolve()
        cls.red = cls._clip("red's source.mp4", "red", audio=True)
        cls.blue = cls._clip("blue source.mp4", "blue", audio=True)
        cls.silent = cls._clip("silent.mp4", "red", audio=False)
        cls.tall = cls._clip("tall.mp4", "red", audio=True, size="90x160")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _clip(cls, name, color, *, audio, size="160x90"):
        path = cls.root / name
        command = [shutil.which("ffmpeg"), "-nostdin", "-v", "error",
                   "-f", "lavfi", "-i", f"color=c={color}:s={size}:r=10:d=1"]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1",
                        "-c:a", "aac"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", "1", str(path)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError("Cannot create offline fixture: " + result.stderr)
        return path

    def test_full_decode_and_metadata(self):
        report = inspect_video(self.red, expected_duration=1, expected_aspect="16:9", require_audio=True)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["decode_ok"])
        self.assertEqual(report["decoded_frames"], 10)
        self.assertEqual((report["width"], report["height"], report["fps"]), (160, 90, 10))
        self.assertTrue(report["has_audio"])

    def test_expectation_failures_are_not_semantic_acceptance(self):
        report = inspect_video(self.silent, expected_duration=5, expected_aspect="9:16", require_audio=True)
        self.assertFalse(report["ok"])
        self.assertTrue(report["decode_ok"])
        self.assertEqual(len(report["issues"]), 3)
        self.assertTrue(report["limitations"])

    def test_invalid_media_and_invalid_options(self):
        broken = self.root / "broken.mp4"
        broken.write_bytes(b"not a video")
        self.assertFalse(inspect_video(broken)["ok"])
        self.assertFalse(inspect_video(self.root / "missing.mp4")["ok"])
        for duration in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(MediaError):
                inspect_video(self.red, expected_duration=duration)
        with self.assertRaises(MediaError):
            inspect_video(self.red, expected_aspect="0:1")

    def test_sparse_frames_include_true_start_end_and_boundary_neighbors(self):
        report = extract_frames(self.red, self.root / "red_frames", interval_s=0.4,
                                boundary_times=[0.5])
        times = [frame["timestamp_s"] for frame in report["frames"]]
        self.assertEqual(times, [0.0, 0.4, 0.5, 0.8, 0.9])
        self.assertTrue(all(Path(frame["path"]).stat().st_size > 0 for frame in report["frames"]))
        self.assertIn("Sparse frames", report["limitations"][0])
        with self.assertRaisesRegex(MediaError, "empty"):
            extract_frames(self.red, self.root / "red_frames")
        with self.assertRaises(MediaError):
            extract_frames(self.red, self.root / "invalid-frames", interval_s=0)
        with self.assertRaisesRegex(MediaError, "Boundary"):
            extract_frames(self.red, self.root / "outside-frames", boundary_times=[10])

    def test_assembly_preserves_order_audio_and_qc(self):
        report = assemble([self.red, self.blue], self.root / "assembled.mp4")
        self.assertTrue(report["inspection"]["ok"], report)
        self.assertTrue(report["inspection"]["has_audio"])
        self.assertEqual(report["inspection"]["decoded_frames"], 20)
        self.assertAlmostEqual(report["inspection"]["duration_s"], 2, delta=0.1)
        frames = extract_frames(Path(report["path"]), self.root / "assembled_frames",
                                interval_s=2, boundary_times=[1])
        hashes = [hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest()
                  for frame in frames["frames"]]
        # Solid red at the start and before the cut; solid blue after and at end.
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[-1], hashes[-2])
        self.assertNotEqual(hashes[0], hashes[-1])
        self.assertEqual(report["clips"], [str(self.red), str(self.blue)])
        original = Path(report["path"]).read_bytes()
        with self.assertRaisesRegex(MediaError, "new file"):
            assemble([self.red], Path(report["path"]))
        self.assertEqual(Path(report["path"]).read_bytes(), original)

    def test_assembly_rejects_mixed_audio_and_incompatible_video(self):
        for clips, message in (([self.red, self.silent], "Mixed audio"),
                               ([self.red, self.tall], "Incompatible video")):
            output = self.root / "must-not-exist.mp4"
            with self.assertRaisesRegex(MediaError, message):
                assemble(clips, output)
            self.assertFalse(output.exists())
        with self.assertRaises(MediaError):
            assemble([], self.root / "empty.mp4")

    def test_audio_free_assembly_does_not_invent_audio(self):
        report = assemble([self.silent, self.silent], self.root / "silent-assembled.mp4")
        self.assertTrue(report["inspection"]["ok"])
        self.assertFalse(report["inspection"]["has_audio"])


if __name__ == "__main__":
    unittest.main()
