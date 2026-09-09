"""Offline continuity gates using synthetic media, never quality certification."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from anigen.video import runtime
from tests.video import test_runtime as fixtures


class FakeUploader:
    def __init__(self, error=None, before_upload=None):
        self.error, self.before_upload = error, before_upload
        self.calls = []

    def preflight(self):
        pass

    def upload(self, path, mime_type, expected_sha256):
        self.calls.append((path, mime_type, expected_sha256))
        if self.before_upload:
            self.before_upload()
        if self.error:
            raise self.error
        return {"access_url": "https://example.com/uploaded-" + path.name,
                "source_sha256": runtime.digest(path), "source_path": str(path),
                "mime_type": mime_type, "size_bytes": path.stat().st_size}


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = tempfile.TemporaryDirectory(prefix="handoff-fixture-")
        cls.fixture = Path(cls.files.name) / "synthetic.mp4"
        subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=s=160x90:r=2:d=5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(cls.fixture)], check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.files.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="handoff-test-")
        self.addCleanup(self.temporary.cleanup)
        self.runs = Path(self.temporary.name)
        self.client = fixtures.FakeProvider(self.fixture)
        block = patch.object(runtime, "FalProvider", side_effect=AssertionError("No live provider in tests"))
        block.start()
        self.addCleanup(block.stop)
        self.call("init", {"brief": "Synthetic test", "script": "Synthetic continuity test only",
                           "reference_scope_id": "synthetic", "sequence_mode": "reviewed_segments",
                           "shots": [{"id": sid, "goal": "Synthetic state transitions", "duration_s": 5,
                                      "required_checks": ["appearance"], "evidence_ids": ["base"]}
                                     for sid in ("one", "two", "three")], "final_checks": ["appearance"]})
        self.call("evidence", {"id": "base", "reference_scope_id": "synthetic", "kind": "user_reference",
                               "locator": "synthetic://base", "claim": "Test source", "verification": "verified", "verification_basis": "direct_observation",
                               "verification_method": "synthetic_fixture", "observation": "Local fixture only"})
        self.authorize()

    def call(self, action, data=None, live=False, uploader=None):
        if action == "init":
            from tests.video.legacy_fixture import seed_legacy
            return seed_legacy("test", data, self.runs)
        return runtime.dispatch(action, "test", data, runs=self.runs, provider=self.client, live=live, uploader=uploader)

    def authorize(self, ids=None, **extra):
        return self.call("authorize", {"user_instruction": "Synthetic authorization; no external calls",
                                      "max_generations": 8, "max_generated_seconds": 40,
                                      "max_attempts_per_shot": 3, "reference_transfer_ids": ids or [], **extra})

    @staticmethod
    def request(shot="one", **extra):
        return {"shot_id": shot, "endpoint": "minimax/h3-max/text-to-video",
                "arguments": {"prompt": "Synthetic state transition", "prompt_expansion_mode": "balanced",
                              "duration": 5, "resolution": "480P"}, **extra}

    def collect_first(self, accept=True, **extra):
        state = self.call("prepare", self.request(**extra))
        aid = state["attempts"][-1]["id"]
        self.call("submit", {"attempt_id": aid}, live=True)
        self.call("poll", {"attempt_id": aid})
        state = self.call("collect", {"attempt_id": aid})
        if accept:
            self.call("review", {"attempt_id": aid, "report": fixtures.RuntimeTests.report(state["attempts"][-1]["observation"])})
        return aid

    def export(self, aid, tail=1):
        state = self.call("handoff", {"attempt_id": aid, "tail_duration_s": tail,
                                      "continuity_state": {key: "Synthetic fixture: " + key for key in
                                                           ("camera", "character_positions", "motion", "screen_direction", "environment", "next_action")}})
        return list(state["handoffs"].values())[-1]

    def bind(self, package, kind="final_frame", eid="tail", **overrides):
        artifact = package["artifacts"][kind]
        url = "https://example.com/" + eid + (".png" if kind == "final_frame" else ".mp4")
        evidence = {"id": eid, "reference_scope_id": "synthetic", "kind": "user_reference",
                    "locator": "synthetic://uploaded/" + eid, "claim": "Uploaded ending fixture",
                    "verification": "verified", "verification_basis": "direct_observation", "observation": "Synthetic external upload receipt only",
                    "verification_method": "synthetic_uploader", "generation_url": url,
                    "handoff_id": package["id"], "handoff_artifact": kind,
                    "observation_path": artifact["path"], "observation_sha256": artifact["sha256"],
                    "observation_mime_type": artifact["mime_type"]}
        if kind == "tail_video":
            evidence["duration_s"] = artifact["duration_s"]
        self.call("evidence", {**evidence, **overrides})
        self.authorize([eid])
        return url

    def continuation(self, package, url, kind="final_frame", mode="image-to-video", shot="two", eid="tail"):
        data = self.request(shot, handoff_id=package["id"], reference_bindings={url: eid})
        data["endpoint"] = "minimax/h3-max/" + mode
        if mode == "image-to-video":
            data["arguments"]["image_url"] = url
        else:
            data["arguments"]["reference_image_urls" if kind == "final_frame" else "reference_video_urls"] = [url]
        return data

    def test_unaccepted_or_rejected_source_cannot_export_or_prepare_next(self):
        aid = self.collect_first(accept=False)
        with self.assertRaisesRegex(ValueError, "currently accepted"):
            self.export(aid)
        with self.assertRaisesRegex(ValueError, "only next"):
            self.call("prepare", self.request("two"))
        observation = self.call("status")["attempts"][0]["observation"]
        self.call("review", {"attempt_id": aid, "report": fixtures.RuntimeTests.report(observation, "reject")})
        with self.assertRaisesRegex(ValueError, "currently accepted"):
            self.export(aid)
        self.assertEqual(len(self.client.submissions), 1)

    def test_export_retains_actual_ending_and_prepare_requires_payload_binding(self):
        aid = self.collect_first()
        with self.assertRaisesRegex(ValueError, "requires handoff_id"):
            self.call("prepare", self.request("two"))
        package = self.export(aid)
        source = self.call("status")["attempts"][0]["observation"]
        self.assertEqual(package["artifacts"]["final_frame"]["sha256"], source["frames"][-1]["sha256"])
        self.assertEqual(package["artifacts"]["final_frame"]["source_timestamp_s"], 4.5)
        self.assertEqual(package["ending"]["requested_window_s"], {"start": 4, "end": 5})
        self.assertTrue(Path(package["manifest_path"]).is_file())
        with self.assertRaisesRegex(ValueError, "payload must use"):
            self.call("prepare", self.request("two", handoff_id=package["id"]))
        self.assertEqual(len(self.client.submissions), 1)

    def test_final_frame_i2v_and_r2v_both_use_exact_uploaded_reference(self):
        package = self.export(self.collect_first())
        url = self.bind(package)
        data = self.continuation(package, url)
        state = self.call("prepare", data)
        aid = state["attempts"][-1]["id"]
        self.call("submit", {"attempt_id": aid}, live=True)
        self.assertEqual(self.client.submissions[-1][1]["image_url"], url)
        self.call("poll", {"attempt_id": aid})
        observed = self.call("collect", {"attempt_id": aid})["attempts"][-1]["observation"]
        self.call("review", {"attempt_id": aid, "report": fixtures.RuntimeTests.report(observed)})
        next_package = self.export(aid)
        next_url = self.bind(next_package, eid="tail_two")
        following = self.continuation(next_package, next_url, mode="reference-to-video", shot="three", eid="tail_two")
        next_aid = self.call("prepare", following)["attempts"][-1]["id"]
        self.call("submit", {"attempt_id": next_aid}, live=True)
        self.assertEqual(self.client.submissions[-1][1]["reference_image_urls"], [next_url])

    def test_video_only_r2v_requires_at_least_two_second_source_ending(self):
        aid = self.collect_first()
        short = self.export(aid)
        short_url = self.bind(short, "tail_video", eid="short")
        with self.assertRaisesRegex(ValueError, "at least 2 seconds"):
            self.call("prepare", self.continuation(short, short_url, "tail_video", "reference-to-video", eid="short"))
        package = self.export(aid, tail=2)
        url = self.bind(package, "tail_video")
        state = self.call("prepare", self.continuation(package, url, "tail_video", "reference-to-video"))
        self.call("submit", {"attempt_id": state["attempts"][-1]["id"]}, live=True)
        self.assertEqual(self.client.submissions[-1][1]["reference_video_urls"], [url])
        self.assertEqual(package["artifacts"]["tail_video"]["source_end_s"], 5)

    def test_unrelated_frame_or_end_image_only_does_not_establish_continuity(self):
        package = self.export(self.collect_first())
        url = self.bind(package, observation_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "not bound to the exported"):
            self.call("prepare", self.continuation(package, url))
        url = self.bind(package, eid="right")
        data = self.continuation(package, url, eid="right")
        data["arguments"]["end_image_url"] = data["arguments"].pop("image_url")
        with self.assertRaisesRegex(ValueError, "payload must use"):
            self.call("prepare", data)

    def test_exported_artifact_mutation_blocks_before_spending(self):
        package = self.export(self.collect_first())
        url = self.bind(package)
        data = self.continuation(package, url)
        aid = self.call("prepare", data)["attempts"][-1]["id"]
        with Path(package["artifacts"]["final_frame"]["path"]).open("ab") as stream:
            stream.write(b"test mutation")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.call("submit", {"attempt_id": aid}, live=True)
        self.assertEqual(len(self.client.submissions), 1)
        self.assertFalse(self.call("status")["attempts"][-1]["charged"])

    def test_acceptance_review_changed_after_prepare_blocks_submit(self):
        package = self.export(self.collect_first())
        url = self.bind(package)
        aid = self.call("prepare", self.continuation(package, url))["attempts"][-1]["id"]
        # Adversarial local fixture mutation, not an operation on real production state.
        state = self.call("status")
        state["attempts"][0]["reviews"][-1]["continuity_state"] = "A materially different accepted continuation"
        runtime.save(self.runs / "test" / "state.json", state)
        with self.assertRaisesRegex(ValueError, "acceptance review changed"):
            self.call("submit", {"attempt_id": aid}, live=True)
        self.assertEqual(len(self.client.submissions), 1)

    def test_replacement_attempt_cannot_reuse_old_accepted_handoff(self):
        package = self.export(self.collect_first())
        url = self.bind(package)
        self.call("invalidate", {"shot_id": "one", "reason": "Synthetic invalidation"})
        self.collect_first(correction="Synthetic corrected first segment")
        with self.assertRaisesRegex(ValueError, "no longer the accepted"):
            self.call("prepare", self.continuation(package, url))

    def test_downstream_import_cannot_claim_generation_used_handoff(self):
        self.collect_first()
        with self.assertRaisesRegex(ValueError, "first segment only"):
            self.call("import-video", {"shot_id": "two", "media_path": str(self.fixture), "user_instruction": "Synthetic import"})

    def test_upload_needs_live_and_separate_generated_media_transfer_authorization(self):
        package = self.export(self.collect_first())
        uploader = FakeUploader()
        data = {"handoff_id": package["id"], "artifact": "final_frame"}
        with self.assertRaisesRegex(ValueError, "--live"):
            self.call("upload-handoff", data, uploader=uploader)
        with self.assertRaisesRegex(ValueError, "not authorized"):
            self.call("upload-handoff", data, live=True, uploader=uploader)
        self.assertEqual(uploader.calls, [])
        self.assertFalse(self.call("status").get("handoff_uploads"))

    def test_upload_reserves_first_then_binds_evidence_and_reuses_without_second_upload(self):
        package = self.export(self.collect_first())
        self.authorize(allow_generated_handoff_transfer=True, max_handoff_uploads=1)
        def assert_reserved():
            state = json.loads((self.runs / "test" / "state.json").read_text())
            self.assertEqual(state["handoff_uploads"]["handoff_upload_0001"]["status"], "upload_unknown")
            self.assertNotIn("handoff_upload_0001_final_frame", state["evidence"])
        uploader = FakeUploader(before_upload=assert_reserved)
        data = {"handoff_id": package["id"], "artifact": "final_frame"}
        state = self.call("upload-handoff", data, live=True, uploader=uploader)
        record = state["handoff_uploads"]["handoff_upload_0001"]
        evidence = state["evidence"][record["evidence_id"]]
        self.assertEqual(record["status"], "completed")
        self.assertIn(evidence["id"], state["authorization"]["reference_transfer_ids"])
        self.assertEqual(evidence["observation_sha256"], package["artifacts"]["final_frame"]["sha256"])
        self.call("upload-handoff", data, live=True, uploader=uploader)
        self.assertEqual(len(uploader.calls), 1)
        request = self.continuation(package, evidence["generation_url"], eid=evidence["id"])
        aid = self.call("prepare", request)["attempts"][-1]["id"]
        self.call("submit", {"attempt_id": aid}, live=True)
        self.assertEqual(len(self.client.submissions), 2)

    def test_uncertain_upload_blocks_duplicate_even_after_authorization_update(self):
        package = self.export(self.collect_first())
        self.authorize(allow_generated_handoff_transfer=True, max_handoff_uploads=2)
        uploader = FakeUploader(error=TimeoutError("Synthetic unknown upload"))
        data = {"handoff_id": package["id"], "artifact": "final_frame"}
        with self.assertRaises(TimeoutError):
            self.call("upload-handoff", data, live=True, uploader=uploader)
        state = self.call("status")
        self.assertEqual(state["handoff_uploads"]["handoff_upload_0001"]["status"], "upload_unknown")
        self.authorize(allow_generated_handoff_transfer=True, max_handoff_uploads=3)
        uploader.error = None
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            self.call("upload-handoff", data, live=True, uploader=uploader)
        self.assertEqual(len(uploader.calls), 1)

    def test_known_upload_failure_retry_needs_reason_and_keeps_cumulative_budget(self):
        package = self.export(self.collect_first())
        self.authorize(allow_generated_handoff_transfer=True, max_handoff_uploads=1)
        failure = RuntimeError("Synthetic pre-upload failure")
        failure.outcome_unknown = False
        uploader = FakeUploader(error=failure)
        data = {"handoff_id": package["id"], "artifact": "final_frame"}
        with self.assertRaises(RuntimeError):
            self.call("upload-handoff", data, live=True, uploader=uploader)
        self.assertEqual(self.call("status")["handoff_uploads"]["handoff_upload_0001"]["status"], "failed")
        with self.assertRaisesRegex(ValueError, "requires a reason"):
            self.call("upload-handoff", data, live=True, uploader=uploader)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.call("upload-handoff", {**data, "retry_reason": "Synthetic explicit retry"}, live=True, uploader=uploader)
        self.assertEqual(len(uploader.calls), 1)
