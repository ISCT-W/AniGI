"""Offline runtime behavior; review reports attest synthetic fixtures only.

These reports test state transitions, never establish real semantic video quality.
Every provider call is fake; real FFmpeg operates only on temporary local fixtures.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from anigen.video import media, runtime
from anigen.video.provider import ProviderError, SubmissionUncertain


class FakeProvider:
    def __init__(self, fixture, *, submit_error=None, outcome=None):
        self.fixture = fixture
        self.submit_error = submit_error
        self.outcome = outcome
        self.submissions = []
        self.downloads = 0

    def submit(self, endpoint, arguments):
        self.submissions.append((endpoint, arguments))
        if self.submit_error:
            raise self.submit_error
        return self.receipt()

    def receipt(self):
        request_id = f"fake-{len(self.submissions)}"
        base = f"https://queue.fal.run/minimax/h3-max/requests/{request_id}"
        return {"request_id": request_id, "status_url": base + "/status",
                "response_url": base, "cancel_url": base + "/cancel"}

    def status(self, receipt):
        return {"status": "COMPLETED", "request_id": receipt["request_id"]}

    def result(self, receipt):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome if self.outcome is not None else {"video": {"url": "https://fal.media/fake.mp4"}}

    def download(self, url, path):
        self.downloads += 1
        shutil.copyfile(self.fixture, path)
        return path


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe unavailable")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = tempfile.TemporaryDirectory(prefix="gpt-agent-runtime-fixture-")
        cls.fixture = Path(cls.fixtures.name).resolve() / "synthetic.mp4"
        command = [shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-f", "lavfi",
                   "-i", "color=c=red:s=160x90:r=2:d=5", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-t", "5", str(cls.fixture)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError("Cannot create synthetic offline fixture: " + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gpt-agent-runtime-test-")
        self.addCleanup(self.temporary.cleanup)
        self.runs = Path(self.temporary.name).resolve()
        self.client = FakeProvider(self.fixture)
        # Any accidentally unmocked real-provider construction fails before network use.
        self.no_network = patch.object(runtime, "FalProvider", side_effect=AssertionError("No real provider in offline tests"))
        self.real_provider = self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def call(self, action, data=None, *, run="test", live=False, client=True):
        if action == "init":
            from tests.video.legacy_fixture import seed_legacy
            return seed_legacy(run, data, self.runs)
        return runtime.dispatch(action, run, data, live=live, runs=self.runs,
                                provider=self.client if client else None)

    def setup_run(self, *, run="test", shots=1, authorize=True, limits=None):
        self.call("init", {
            "brief": "Offline synthetic fixture", "script": "A static red frame; no real generated content",
            "reference_scope_id": "synthetic-project", "final_checks": ["appearance"],
            "shots": [{"id": f"shot_{i + 1}", "goal": "Exercise state transitions",
                       "duration_s": 5, "aspect_ratio": "16:9", "required_checks": ["appearance"],
                       "evidence_ids": ["reference"]} for i in range(shots)],
        }, run=run)
        self.add_evidence("reference", run=run)
        if authorize:
            self.authorize(run=run, **(limits or {}))

    def add_evidence(self, eid, *, run="test", url=None, duration=None):
        evidence = {"id": eid, "reference_scope_id": "synthetic-project", "kind": "user_reference",
                    "locator": "synthetic://fixture/" + eid, "claim": "Synthetic test fixture only",
                    "verification": "verified", "verification_basis": "direct_observation", "observation": "Fixture built locally for tests",
                    "verification_method": "synthetic_fixture_builder"}
        if url is not None:
            evidence["generation_url"] = url
        if duration is not None:
            evidence["duration_s"] = duration
        self.call("evidence", evidence, run=run)

    def authorize(self, *, run="test", **overrides):
        data = {"user_instruction": "Synthetic test authorization, no real generation",
                "max_generations": 10, "max_generated_seconds": 50,
                "max_attempts_per_shot": 3, "reference_transfer_ids": []}
        self.call("authorize", {**data, **overrides}, run=run)

    def prepare(self, *, run="test", shot="shot_1", correction=None, references=None, bindings=None):
        args = {"prompt": "Synthetic fixture test", "prompt_expansion_mode": "balanced",
                "duration": 5, "resolution": "480P", "aspect_ratio": "16:9"}
        data = {"shot_id": shot, "endpoint": "minimax/h3-max/text-to-video", "arguments": args}
        if correction:
            data["correction"] = correction
        if references is not None:
            data["endpoint"] = "minimax/h3-max/reference-to-video"
            args.update(references)
            data["reference_bindings"] = bindings or {}
        state = self.call("prepare", data, run=run)
        return state["attempts"][-1]["id"]

    def collect(self, *, run="test", shot="shot_1", correction=None):
        aid = self.prepare(run=run, shot=shot, correction=correction)
        self.call("submit", {"attempt_id": aid}, run=run, live=True)
        self.call("poll", {"attempt_id": aid}, run=run)
        state = self.call("collect", {"attempt_id": aid}, run=run)
        self.assertEqual(state["attempts"][-1]["status"], "review_pending")
        return aid, state["attempts"][-1]["observation"]

    @staticmethod
    def report(observation, decision="accept"):
        result = {"media_sha256": observation["media_sha256"],
                  "observed_frames": [f["path"] for f in observation["frames"]],
                  "summary": "Synthetic fixture assertion only; no semantic-quality assertion",
                  "observation_mode": "sampled_frames", "decision": decision,
                  "checks": {"appearance": {"status": "pass" if decision == "accept" else "fail",
                             "reason": "Known static-color fixture used only to test runtime behavior"}},
                  "continuity_state": "Synthetic fixture continuation"}
        if decision == "reject":
            result["correction"] = "Change the synthetic fixture prompt for the retry test"
        return result

    def accept(self, *, run="test", shot="shot_1"):
        aid, observation = self.collect(run=run, shot=shot)
        self.call("review", {"attempt_id": aid, "report": self.report(observation)}, run=run)
        return aid

    def test_complete_flow_persists_artifacts_and_final_acceptance(self):
        self.setup_run(shots=2)
        first = self.accept(shot="shot_1")
        second = self.accept(shot="shot_2")
        assembled = self.call("assemble")
        observation = assembled["final"]["observation"]
        self.assertTrue(observation["qc"]["ok"])
        self.assertAlmostEqual(observation["qc"]["duration_s"], 10, delta=0.1)
        self.assertEqual(observation["qc"]["decoded_frames"], 20)
        self.call("final-review", self.report(observation))
        persisted = self.call("status", client=False)
        self.assertEqual(persisted["accepted"], {"shot_1": first, "shot_2": second})
        self.assertEqual(persisted["final"]["status"], "accepted")
        self.assertFalse(persisted["authorization"]["active"])
        self.assertEqual(len(self.client.submissions), 2)
        self.assertEqual(self.client.downloads, 2)
        self.assertEqual(persisted["events"][-1]["kind"], "final_reviewed")

    def test_rejection_requires_correction_then_allows_bounded_retry(self):
        self.setup_run()
        aid, observation = self.collect()
        self.call("review", {"attempt_id": aid, "report": self.report(observation, "reject")})
        with self.assertRaisesRegex(ValueError, "correction"):
            self.prepare()
        retry, observation = self.collect(correction="Apply synthetic fixture correction")
        state = self.call("review", {"attempt_id": retry, "report": self.report(observation)})
        self.assertEqual([a["status"] for a in state["attempts"]], ["rejected", "accepted"])
        self.assertEqual(state["accepted"], {"shot_1": retry})

    def test_submit_needs_live_and_authorization_and_cannot_repeat_post(self):
        self.setup_run(authorize=False)
        aid = self.prepare()
        with self.assertRaisesRegex(ValueError, "--live"):
            self.call("submit", {"attempt_id": aid})
        with self.assertRaisesRegex(ValueError, "not authorized"):
            self.call("submit", {"attempt_id": aid}, live=True)
        self.assertEqual(self.client.submissions, [])
        self.assertFalse(self.call("status")["attempts"][0]["charged"])
        self.authorize()
        self.call("submit", {"attempt_id": aid}, live=True)
        with self.assertRaisesRegex(ValueError, "never repeat POST"):
            self.call("submit", {"attempt_id": aid}, live=True)
        self.assertEqual(len(self.client.submissions), 1)

    def test_independent_count_seconds_and_per_shot_limits(self):
        settings = [("count", {"max_generations": 1}, "generation-count"),
                    ("seconds", {"max_generated_seconds": 5}, "generated-seconds"),
                    ("shot", {"max_attempts_per_shot": 1}, "shot attempt")]
        for run, limits, error in settings:
            with self.subTest(limit=run):
                self.setup_run(run=run, limits=limits)
                self.client.outcome = {"error": "Synthetic model failure"}
                first = self.prepare(run=run)
                self.call("submit", {"attempt_id": first}, run=run, live=True)
                self.call("poll", {"attempt_id": first}, run=run)
                retry = self.prepare(run=run, correction="Retry synthetic model failure")
                before = len(self.client.submissions)
                with self.assertRaisesRegex(ValueError, error):
                    self.call("submit", {"attempt_id": retry}, run=run, live=True)
                state = self.call("status", run=run)
                self.assertEqual(len(self.client.submissions), before)
                self.assertFalse(state["attempts"][-1]["charged"])

    def test_uncertain_submission_is_reserved_and_requires_receipt_recovery(self):
        self.setup_run()
        aid = self.prepare()
        self.client.submit_error = SubmissionUncertain("Synthetic connection interruption")
        with self.assertRaises(SubmissionUncertain):
            self.call("submit", {"attempt_id": aid}, live=True)
        state = self.call("status")
        self.assertEqual(state["attempts"][0]["status"], "submission_unknown")
        self.assertTrue(state["attempts"][0]["charged"])
        with self.assertRaises(ValueError):
            self.call("submit", {"attempt_id": aid}, live=True)
        with self.assertRaises(ValueError):
            self.prepare(correction="Must not create duplicate remote work")
        self.call("recover", {"attempt_id": aid, "receipt": self.client.receipt(),
                              "recovery_evidence": "Synthetic provider dashboard receipt"})
        self.call("poll", {"attempt_id": aid})
        self.assertEqual(len(self.client.submissions), 1)
        self.assertEqual(self.call("status")["attempts"][0]["status"], "completed")

    def test_r2v_plural_urls_need_transfer_permission_and_exact_evidence_binding(self):
        self.setup_run()
        image_url, video_url = "https://example.com/image.png", "https://example.com/reference.mp4"
        self.add_evidence("image", url=image_url)
        self.add_evidence("clip", url=video_url, duration=5)
        aid = self.prepare(references={"reference_image_urls": [image_url], "reference_video_urls": [video_url]},
                           bindings={image_url: "image", video_url: "clip"})
        with self.assertRaisesRegex(ValueError, "not authorized"):
            self.call("submit", {"attempt_id": aid}, live=True)
        self.authorize(reference_transfer_ids=["image"])
        with self.assertRaisesRegex(ValueError, "not authorized"):
            self.call("submit", {"attempt_id": aid}, live=True)
        self.assertEqual(len(self.client.submissions), 0)
        self.authorize(reference_transfer_ids=["image", "clip"])
        self.call("submit", {"attempt_id": aid}, live=True)
        self.assertEqual(len(self.client.submissions), 1)
        self.setup_run(run="mismatch")
        self.add_evidence("wrong", run="mismatch", url="https://example.com/other.png")
        self.authorize(run="mismatch", reference_transfer_ids=["wrong"])
        aid = self.prepare(run="mismatch", references={"reference_image_urls": [image_url]}, bindings={image_url: "wrong"})
        with self.assertRaisesRegex(ValueError, "not bound"):
            self.call("submit", {"attempt_id": aid}, run="mismatch", live=True)
        self.assertFalse(self.call("status", run="mismatch")["attempts"][0]["charged"])

    def test_model_error_and_http422_become_failed_not_collectable(self):
        for run, outcome in (("error", {"error": "Synthetic model failure"}),
                             ("unprocessable", ProviderError("Synthetic failure", http_status=422))):
            with self.subTest(outcome=run):
                self.setup_run(run=run)
                self.client.outcome = outcome
                aid = self.prepare(run=run)
                self.call("submit", {"attempt_id": aid}, run=run, live=True)
                state = self.call("poll", {"attempt_id": aid}, run=run)
                self.assertEqual(state["attempts"][0]["status"], "failed")
                self.assertTrue(state["attempts"][0]["charged"])
                with self.assertRaisesRegex(ValueError, "not complete"):
                    self.call("collect", {"attempt_id": aid}, run=run)
        self.assertEqual(self.client.downloads, 0)

    def test_changed_media_or_frame_cannot_reuse_review(self):
        for run, target in (("video", "media_path"), ("frame", "frame")):
            with self.subTest(target=target):
                self.setup_run(run=run)
                aid, observation = self.collect(run=run)
                report = self.report(observation)
                path = Path(observation["frames"][0]["path"] if target == "frame" else observation[target])
                with path.open("ab") as stream:
                    stream.write(b"changed after observation")
                with self.assertRaisesRegex(ValueError, "changed since observation"):
                    self.call("review", {"attempt_id": aid, "report": report}, run=run)
                self.assertEqual(self.call("status", run=run)["attempts"][0]["status"], "review_pending")

    def test_sampled_frames_cannot_clear_temporal_or_audio_checks(self):
        self.setup_run()
        aid, observation = self.collect()
        for check in ("temporal", "audio"):
            with self.subTest(check=check):
                report = self.report(observation)
                report["checks"][check] = {"status": "pass", "reason": "Intentionally unsupported test assertion"}
                with self.assertRaises(ValueError):
                    self.call("review", {"attempt_id": aid, "report": report})
        self.assertEqual(self.call("status")["accepted"], {})

    def test_invalidation_discards_downstream_acceptance_and_preserves_final_history(self):
        self.setup_run(shots=2)
        first = self.accept(shot="shot_1")
        self.accept(shot="shot_2")
        old_final = self.call("assemble")["final"]
        state = self.call("invalidate", {"shot_id": "shot_2", "reason": "Synthetic continuity correction"})
        self.assertEqual(state["accepted"], {"shot_1": first})
        self.assertEqual([a["status"] for a in state["attempts"]], ["accepted", "superseded"])
        self.assertIsNone(state["final"])
        self.assertEqual(state["final_history"], [old_final])
        state = self.call("invalidate", {"shot_id": "shot_1", "reason": "Reset synthetic opening"})
        self.assertEqual(state["accepted"], {})
        self.assertTrue(all(a["status"] == "superseded" for a in state["attempts"]))

    def test_missing_or_invalid_key_does_not_reserve_or_construct_provider(self):
        self.setup_run()
        aid = self.prepare()
        for key in ("", "invalid\nkey", "invalid\x7fkey"):
            with self.subTest(key_form="empty" if not key else "control_character"):
                with patch.dict("os.environ", {"FAL_KEY": key}):
                    with self.assertRaisesRegex(ValueError, "FAL_KEY missing"):
                        self.call("submit", {"attempt_id": aid}, live=True, client=False)
        state = self.call("status")
        self.assertEqual(state["attempts"][0]["status"], "prepared")
        self.assertFalse(state["attempts"][0]["charged"])
        self.assertNotIn("submission_reserved", [e["kind"] for e in state["events"]])
        self.real_provider.assert_not_called()

    def test_check_names_cannot_bypass_observation_requirements(self):
        self.setup_run()
        state = self.call("status")
        for name in ("动作连贯性", "audio_sync", "unknown_criterion"):
            with self.subTest(name=name):
                plan = state["plan"]
                plan["shots"][0]["required_checks"] = [name]
                with self.assertRaisesRegex(ValueError, "canonical"):
                    runtime.validate_plan(plan)
                observation = {"media_sha256": "synthetic", "frames": [{"path": "synthetic-frame"}]}
                report = self.report(observation)
                report["checks"] = {name: {"status": "pass", "reason": "Deliberately invalid test assertion"}}
                with self.assertRaisesRegex(ValueError, "unknown"):
                    runtime.check_review(report, observation, [name])
        for name in ("temporal", "camera", "pacing", "narrative", "audio", "lip_sync"):
            with self.subTest(name=name):
                observation = {"media_sha256": "synthetic", "frames": [{"path": "synthetic-frame"}]}
                report = self.report(observation)
                report["checks"] = {name: {"status": "pass", "reason": "Deliberately unsupported test assertion"}}
                with self.assertRaises(ValueError):
                    runtime.check_review(report, observation, [name])

    def test_interrupted_collect_and_assembly_resume_without_duplicate_submission(self):
        self.setup_run()
        aid = self.prepare()
        self.call("submit", {"attempt_id": aid}, live=True)
        self.call("poll", {"attempt_id": aid})
        real_extract = media.extract_frames

        def interrupted_extract(*args, **kwargs):
            real_extract(*args, **kwargs)
            raise RuntimeError("Synthetic crash after writing extracted frames")

        with patch.object(runtime.media, "extract_frames", side_effect=interrupted_extract):
            with self.assertRaisesRegex(RuntimeError, "Synthetic crash"):
                self.call("collect", {"attempt_id": aid})
        self.assertEqual(self.call("status")["attempts"][0]["status"], "completed")
        state = self.call("collect", {"attempt_id": aid})
        self.assertEqual(self.client.downloads, 1)
        self.call("review", {"attempt_id": aid, "report": self.report(state["attempts"][0]["observation"])})
        real_assemble = media.assemble

        def interrupted_assemble(*args, **kwargs):
            real_assemble(*args, **kwargs)
            raise RuntimeError("Synthetic crash after writing assembled video")

        with patch.object(runtime.media, "assemble", side_effect=interrupted_assemble):
            with self.assertRaisesRegex(RuntimeError, "Synthetic crash"):
                self.call("assemble")
        self.assertIsNone(self.call("status")["final"])
        final = self.call("assemble")["final"]
        self.assertTrue(final["observation"]["qc"]["ok"])
        self.assertEqual(final["status"], "review_pending")
        self.assertEqual(len(self.client.submissions), 1)


if __name__ == "__main__":
    unittest.main()
