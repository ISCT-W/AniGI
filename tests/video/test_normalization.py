"""Real FFmpeg counterexamples and fake-provider provenance gates; no network."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from anigen.video import media, normalization, runtime
from tests.video import test_runtime as fixtures


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = tempfile.TemporaryDirectory(prefix="normalize-fixture-")
        cls.root = Path(cls.files.name)
        for name, duration, size, sound, offset in [
            ("bad", 5.125, "336x192", True, 0),
            ("good", 5, "320x180", False, 0),
            ("mute_shifted", 5.125, "336x192", False, 2),
            ("short", 4.5, "336x192", False, 0),
            ("long", 5.5, "336x192", False, 0),
        ]:
            cmd = [shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                   f"testsrc2=s={size}:r=24:d={duration}"]
            if sound:
                cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}", "-c:a", "aac"]
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(duration)]
            if offset:
                cmd += ["-output_ts_offset", str(offset)]
            subprocess.run([*cmd, str(cls.root / f"{name}.mp4")], check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.files.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="normalize-test-")
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name)
        self.provider = fixtures.FakeProvider(self.root / "bad.mp4")
        block = patch.object(runtime, "FalProvider", side_effect=AssertionError("No live provider"))
        block.start()
        self.addCleanup(block.stop)
        self.call("init", {"brief": "Synthetic normalization", "script": "Offline provenance test",
                           "reference_scope_id": "synthetic", "sequence_mode": "reviewed_segments",
                           "shots": [{"id": sid, "goal": "Synthetic", "duration_s": 5, "aspect_ratio": "16:9",
                                      "required_checks": ["appearance"], "evidence_ids": ["base"]} for sid in ("one", "two")],
                           "final_checks": ["appearance"]})
        self.call("evidence", {"id": "base", "reference_scope_id": "synthetic", "kind": "user_reference",
                               "locator": "synthetic://base", "claim": "Synthetic", "verification": "verified", "verification_basis": "direct_observation",
                               "verification_method": "local_fixture", "observation": "Synthetic fixture only"})
        self.call("authorize", {"user_instruction": "Synthetic authorization only", "max_generations": 6,
                                "max_generated_seconds": 30, "max_attempts_per_shot": 3, "reference_transfer_ids": ["base"]})

    def call(self, action, data=None, live=False):
        if action == "init":
            from tests.video.legacy_fixture import seed_legacy
            return seed_legacy("test", data, self.runs)
        return runtime.dispatch(action, "test", data, runs=self.runs, provider=self.provider, live=live)

    def collect(self, shot="one", **extra):
        request = {"shot_id": shot, "endpoint": "minimax/h3-max/text-to-video",
                   "arguments": {"prompt": "Synthetic", "prompt_expansion_mode": "balanced", "duration": 5,
                                 "aspect_ratio": "16:9", "resolution": "480P"}, **extra}
        state = self.call("prepare", request)
        aid = state["attempts"][-1]["id"]
        self.call("submit", {"attempt_id": aid}, live=True)
        self.call("poll", {"attempt_id": aid})
        return self.call("collect", {"attempt_id": aid})["attempts"][-1]

    def normalize(self, aid):
        return self.call("normalize-video", {"attempt_id": aid, "reason": "Synthetic 7:4 and 0.125-second excess"})

    def test_real_counterexample_retains_original_metadata_qc_and_charge(self):
        original = self.collect()
        self.assertEqual(original["status"], "failed")
        self.assertAlmostEqual(original["observation"]["qc"]["duration_s"], 5.125)
        self.assertFalse(original["observation"]["qc"]["ok"])
        state = self.normalize(original["id"])
        derived = state["attempts"][-1]
        record = derived["normalizations"][0]
        self.assertEqual(derived["id"], original["id"])
        self.assertEqual(derived["status"], "review_pending")
        self.assertEqual(len(state["attempts"]), 1)
        self.assertEqual(len(self.provider.submissions), 1)
        self.assertTrue(derived["charged"])
        for key in ("endpoint", "arguments", "receipt", "result", "charged"):
            self.assertEqual(derived[key], original[key])
            self.assertEqual(record["generation_provenance"][key], original[key])
        self.assertEqual(record["original_observation"], original["observation"])
        self.assertEqual(runtime.digest(record["source_path"]), original["observation"]["media_sha256"])
        self.assertNotEqual(record["source_path"], record["output_path"])
        self.assertEqual(record["tail_removed_s"], 0.125)
        self.assertEqual(record["frames_added"], 0)
        self.assertEqual(record["output_frame_count"], 120)
        self.assertTrue(record["source_metadata"]["streams"])
        self.assertTrue(record["output_metadata"]["streams"])
        self.assertEqual(record["output_qc"]["duration_s"], 5)
        self.assertTrue(record["output_qc"]["has_audio"])
        self.assertEqual(record["canvas"]["width"] / record["canvas"]["height"], 16 / 9)
        self.assertLess(record["fit"]["width"], record["canvas"]["width"])
        self.assertTrue(derived["observation"]["frames"])
        saved = json.loads(Path(record["manifest_path"]).read_text())
        self.assertEqual(saved["generation_provenance"]["receipt"], original["receipt"])
        self.assertEqual(runtime.digest(record["manifest_path"]), record["manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "already normalized"):
            self.normalize(original["id"])

    def test_muted_nonzero_origin_is_zeroed_without_filling(self):
        self.provider.fixture = self.root / "mute_shifted.mp4"
        original = self.collect()
        self.assertAlmostEqual(original["observation"]["qc"]["video_start_s"], 2)
        record = self.normalize(original["id"])["attempts"][-1]["normalizations"][0]
        self.assertFalse(record["output_qc"]["has_audio"])
        self.assertAlmostEqual(record["output_qc"]["video_start_s"], 0)
        self.assertEqual(record["output_qc"]["decoded_frames"], 120)

    def test_short_and_large_tail_are_rejected_without_state_change(self):
        for name, message in [("short", "must not extend"), ("long", "at most 0.2")]:
            with self.subTest(name=name):
                destination = self.runs / f"{name}-output.mp4"
                with self.assertRaisesRegex(media.MediaError, message):
                    normalization.normalize(self.root / f"{name}.mp4", destination, duration_s=5, aspect_ratio="16:9")
                self.assertFalse(destination.exists())

    def test_existing_observation_or_review_records_block_derivation(self):
        attempt = self.collect()
        original = self.call("status")
        for collection in ("av_observations", "human_reviews", "inspection_windows", "handoffs"):
            with self.subTest(collection=collection):
                state = json.loads(json.dumps(original))
                record = ({"request": {"target_id": attempt["id"]}, "status": "in_progress"}
                          if collection == "av_observations" else {"target_id": attempt["id"]})
                state[collection] = {"synthetic_record": record}
                runtime.save(self.runs / "test/state.json", state)
                with self.assertRaisesRegex(ValueError, "must precede"):
                    self.normalize(attempt["id"])
        for field, value, message in [("reviews", [{"decision": "needs_review"}], "reviewed"),
                                      ("imported", True, "fal result")]:
            state = json.loads(json.dumps(original))
            state["attempts"][-1][field] = value
            runtime.save(self.runs / "test/state.json", state)
            with self.assertRaisesRegex(ValueError, message):
                self.normalize(attempt["id"])

    def test_old_attempt_is_not_eligible(self):
        first = self.collect()
        self.collect(correction="Synthetic second attempt after failed technical QC")
        with self.assertRaisesRegex(ValueError, "current attempt"):
            self.normalize(first["id"])

    def test_original_manifest_and_generation_binding_cannot_drift_before_review(self):
        aid = self.collect()["id"]
        state = self.normalize(aid)
        target = state["attempts"][-1]
        record = target["normalizations"][0]
        review = {"attempt_id": aid, "report": fixtures.RuntimeTests.report(target["observation"])}
        for file_key, message in [("source_path", "media changed"), ("manifest_path", "manifest changed")]:
            with self.subTest(file=file_key):
                path = Path(record[file_key])
                old = path.read_bytes()
                path.write_bytes(old + b"\n")
                with self.assertRaisesRegex(ValueError, message):
                    self.call("review", review)
                path.write_bytes(old)
        changed = json.loads(json.dumps(state))
        changed["attempts"][-1]["arguments"]["prompt"] = "Different hidden generation request"
        runtime.save(self.runs / "test/state.json", changed)
        with self.assertRaisesRegex(ValueError, "generation request"):
            self.call("review", review)
        runtime.save(self.runs / "test/state.json", state)
        accepted = self.call("review", review)
        self.assertEqual(accepted["accepted"]["one"], aid)
        runtime.verify_checkpoints(accepted)

    def test_second_segment_preserves_and_rechecks_actual_handoff(self):
        self.provider.fixture = self.root / "good.mp4"
        first = self.collect()
        self.call("review", {"attempt_id": first["id"], "report": fixtures.RuntimeTests.report(first["observation"])})
        with self.assertRaisesRegex(ValueError, "current attempt"):
            self.normalize(first["id"])
        with self.assertRaisesRegex(ValueError, "requires handoff_id"):
            self.collect("two")
        state = self.call("handoff", {"attempt_id": first["id"], "tail_duration_s": 1,
                                      "continuity_state": {k: "Synthetic" for k in ("camera", "character_positions", "motion", "screen_direction", "environment", "next_action")}})
        package = list(state["handoffs"].values())[-1]
        artifact = package["artifacts"]["final_frame"]
        url = "https://example.com/synthetic-tail.png"
        self.call("evidence", {"id": "tail", "reference_scope_id": "synthetic", "kind": "user_reference",
                               "locator": "synthetic://tail", "claim": "Synthetic", "verification": "verified", "verification_basis": "direct_observation",
                               "verification_method": "fake_upload", "observation": "Synthetic receipt", "generation_url": url,
                               "handoff_id": package["id"], "handoff_artifact": "final_frame",
                               "observation_path": artifact["path"], "observation_sha256": artifact["sha256"],
                               "observation_mime_type": artifact["mime_type"]})
        auth = self.call("status")["authorization"]
        self.call("authorize", {**auth, "reference_transfer_ids": ["base", "tail"]})
        self.provider.fixture = self.root / "bad.mp4"
        second = self.collect("two", endpoint="minimax/h3-max/image-to-video", handoff_id=package["id"],
                              reference_bindings={url: "tail"}, arguments={"prompt": "Synthetic continuation", "prompt_expansion_mode": "balanced",
                                                                          "duration": 5, "resolution": "480P", "image_url": url})
        original = self.call("status")
        tampered = json.loads(json.dumps(original))
        tampered["attempts"][-1].pop("handoff_id")
        runtime.save(self.runs / "test/state.json", tampered)
        with self.assertRaisesRegex(ValueError, "requires handoff_id"):
            self.normalize(second["id"])
        runtime.save(self.runs / "test/state.json", original)
        state = self.normalize(second["id"])
        target = state["attempts"][-1]
        self.assertEqual(target["handoff_id"], package["id"])
        self.assertEqual(target["arguments"]["image_url"], url)
        self.assertEqual(target["reference_bindings"], {url: "tail"})
        self.assertEqual(target["normalizations"][0]["generation_provenance"]["handoff_id"], package["id"])
        self.assertEqual(target["normalizations"][0]["generation_provenance"]["receipt"], second["receipt"])
        self.assertEqual(len(state["attempts"]), 2)
        self.assertEqual(len(self.provider.submissions), 2)
        runtime.verify_segment_handoff(state, target)


if __name__ == "__main__":
    unittest.main()
