"""Offline integration using synthetic video and fake observations; no network."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from anigen.video import runtime, observation as av
from anigen.video.gemini_observer import (
    DEFAULT_MODEL, PROMPT_VERSION, BLIND_PROMPT_VERSION, BLIND_SYSTEM_PROMPT_SHA256,
    GeminiObserver, GeminiObserverError,
)


class FakeObserver:
    def __init__(self):
        self.calls = 0
        self.failure = None
        self.findings = []
        self.extra_audio = False
        self.uncertainties = []
        self.model = DEFAULT_MODEL

    def preflight(self):
        return {"api_key_present": True}

    def observe(self, request, directory, callback):
        self.calls += 1
        owned = {"name": "files/fake", "uri": "https://generativelanguage.googleapis.com/v1beta/files/fake", "owned": True, "cleanup_status": "pending"}
        callback(owned)
        if self.failure:
            raise self.failure
        response = {"events": [{"start_s": 0, "end_s": request["uploaded_duration_s"], "description": "Synthetic frame", "evidence": "fixture only"}],
                    "audio_events": [], "dialogue_segments": [], "findings": copy.deepcopy(self.findings), "uncertainties": copy.deepcopy(self.uncertainties)}
        if self.extra_audio:
            response["audio_events"] = copy.deepcopy(response["events"])
        raw = directory / "fake-raw.json"
        raw.write_text(json.dumps(response))
        return {"response": response, "raw_response_path": str(raw), "raw_response_sha256": av.file_hash(raw),
                "remote_files": [{**owned, "cleanup_status": "deleted"}], "cleanup_status": "complete", "usage": None,
                "returned_model": self.model, "requested_model": DEFAULT_MODEL}

    def cleanup(self, files):
        return {"remote_files": [{**f, "cleanup_status": "deleted"} for f in files], "cleanup_status": "complete"}


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
class ObservationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.tmp.name) / "sample.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=160x90:r=10:d=5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cls.video)], check=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.tmp_run = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_run.cleanup)
        self.runs = Path(self.tmp_run.name)
        self.observer = FakeObserver()
        self.call("init", {"brief": "Synthetic test", "script": "Red clip", "reference_scope_id": "synthetic-scope",
                          "source_scope": "synthetic-source", "final_checks": ["motion"],
                          "shots": [{"id": "s1", "goal": "Synthetic test only", "duration_s": 5, "required_checks": ["motion"], "evidence_ids": ["ref"]}]})
        self.call("evidence", {"id": "ref", "reference_scope_id": "synthetic-scope", "source_scope": "synthetic-source", "kind": "user_reference",
                               "verification": "verified", "verification_basis": "direct_observation", "verification_method": "synthetic_fixture", "observation": "red",
                               "locator": "synthetic:test", "claim": "red fixture"})
        self.call("import-video", {"shot_id": "s1", "media_path": str(self.video), "user_instruction": "Offline test fixture only"})
        self.policy()

    def call(self, action, data=None, live=False):
        if action == "init":
            from tests.video.legacy_fixture import seed_legacy
            return seed_legacy("test", data, self.runs)
        return runtime.dispatch(action, "test", data, runs=self.runs, observer=self.observer, live=live)

    def policy(self, qualify=False, **overrides):
        data = {"user_instruction": "Offline fixture policy; no real transfer", "allow_task_spec": True,
                "allow_generated_media": False, "imported_target_ids": ["attempt_0001"], "reference_transfer_ids": [],
                "max_calls": 5, "max_calls_per_target": 4, "max_local_calls_per_target": 1,
                "max_media_seconds": 60, "reserve_final_calls": 1}
        if qualify:
            path = self.runs / "fake-evaluation.json"
            path.write_text('{"synthetic_mock_only":true}')
            data["qualifications"] = [{"model": DEFAULT_MODEL, "returned_model": DEFAULT_MODEL,
                                        "prompt_version": PROMPT_VERSION, "fps": 1, "checks": ["motion"],
                                        "scope": "fixture", "user_confirmation": "Mock qualification only",
                                        "evaluation_report_path": str(path), "evaluation_report_sha256": av.file_hash(path)}]
        return self.call("observation-policy", {**data, **overrides})

    def observe(self, **data):
        return self.call("observe-av", {"target_id": "attempt_0001", "qualification_scope": "fixture", **data}, live=True)

    def report(self, state=None, target="attempt_0001", oid="observation_0001"):
        state = state or self.call("status")
        obj = state["final"] if target == "final" else runtime.get_attempt(state, target)
        obs = obj["observation"]
        return {"media_sha256": obs["media_sha256"], "observed_frames": [f["path"] for f in obs["frames"]],
                "observation_mode": "video", "summary": "Synthetic fake observation contract only",
                "observation_ids": [oid], "checks": {"motion": {"status": "pass", "reason": "fake evidence",
                "evidence_refs": [oid + "/events/0"]}}, "decision": "accept", "continuity_state": "red"}

    def test_live_and_transfer_gates_run_before_call(self):
        with self.assertRaisesRegex(ValueError, "--live"):
            self.call("observe-av", {"target_id": "attempt_0001"})
        self.policy(imported_target_ids=[])
        with self.assertRaisesRegex(ValueError, "not authorized"):
            self.observe()
        self.assertEqual(self.observer.calls, 0)
        self.assertFalse(self.call("status")["av_observations"])

    def test_legacy_playback_string_cannot_clear_motion(self):
        report = self.report()
        report.pop("observation_ids")
        report["playback_evidence"] = "I watched it"
        with self.assertRaises(ValueError):
            self.call("review", {"attempt_id": "attempt_0001", "report": report})
        self.assertEqual(self.call("status")["attempts"][0]["status"], "review_pending")

    def test_receipt_without_quality_qualification_stays_pending(self):
        state = self.observe()
        self.assertEqual(state["av_observations"]["observation_0001"]["validated_capabilities"], [])
        with self.assertRaisesRegex(ValueError, "qualification"):
            self.call("review", {"attempt_id": "attempt_0001", "report": self.report(state)})

    def director_report(self):
        report = self.report()
        report.update(review_authority="codex_director", director_review={
            "reviewer": "codex", "reference_ids": ["ref"],
            "rationale": "Synthetic director judgment, not model certification", "limitations": []})
        report["checks"]["motion"]["director_reason"] = "Synthetic observed sequence supports the fixture goal"
        return report

    def test_director_can_accept_without_model_qualification(self):
        self.policy(review_authority="codex_director")
        self.observe()
        state = self.call("review", {"attempt_id": "attempt_0001", "report": self.director_report()})
        self.assertEqual(state["attempts"][0]["status"], "accepted")
        self.assertEqual(state["av_observations"]["observation_0001"]["validated_capabilities"], [])

    def test_director_requires_policy_reason_references_and_full_comparison(self):
        self.observe()
        with self.assertRaisesRegex(ValueError, "not enabled"):
            self.call("review", {"attempt_id": "attempt_0001", "report": self.director_report()})
        self.policy(review_authority="codex_director")
        for mutate in (lambda r: r["checks"]["motion"].pop("director_reason"),
                       lambda r: r["director_review"].update(reference_ids=[]),
                       lambda r: r.update(observation_ids=[])):
            report = self.director_report(); mutate(report)
            with self.assertRaises(ValueError):
                self.call("review", {"attempt_id": "attempt_0001", "report": report})

    def test_director_still_rejects_unresolved_observer_conflict(self):
        self.policy(review_authority="codex_director")
        self.observer.uncertainties = [{"start_s": 0, "end_s": 5, "description": "Unclear motion", "evidence": "fixture"}]
        self.observe()
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.call("review", {"attempt_id": "attempt_0001", "report": self.director_report()})
        report = self.director_report()
        report["director_conflict_resolutions"] = {"observation_0001/uncertainties/0": {
            "reason": "Synthetic local cross-check resolves fixture ambiguity", "evidence_refs": ["observation_0001/events/0"]}}
        state = self.call("review", {"attempt_id": "attempt_0001", "report": report})
        self.assertEqual(state["attempts"][0]["status"], "accepted")

    def test_director_blocks_static_conflict(self):
        self.policy(review_authority="codex_director")
        self.observer.findings = [{"start_s": 0, "end_s": 5, "check": "appearance", "status": "fail", "description": "Wrong style", "evidence": "fixture"}]
        self.observe()
        report = self.director_report()
        report["checks"]["appearance"] = {"status": "pass", "reason": "Unjustified"}
        with self.assertRaisesRegex(ValueError, "static observer conflict"):
            self.call("review", {"attempt_id": "attempt_0001", "report": report})

    def test_director_resolution_rejects_wrong_check_and_time(self):
        chosen = {"o": {"request": {"source_offset_s": 0}, "response": {"findings": [
            {"check": "identity", "status": "observed", "start_s": 0, "end_s": 0.1}],
            "events": [{"start_s": 0, "end_s": 0.1}]}}}
        finding = {"start_s": 4, "end_s": 5}
        for ref in ("o/findings/0", "o/events/0"):
            with self.assertRaises(ValueError):
                av.director_resolution({"reason": "Unrelated", "evidence_refs": [ref]}, chosen,
                                       "motion", chosen["o"], finding)

    def test_director_uncertainty_can_be_resolved_per_check(self):
        chosen = {"o": {"request": {"source_offset_s": 0}, "response": {
            "events": [{"start_s": 0, "end_s": 5}],
            "dialogue_segments": [{"start_s": 0, "end_s": 5}]}}}
        resolution = {"by_check": {name: {"reason": "Fixture-specific reason", "evidence_refs": [ref]}
                      for name, ref in (("motion", "o/events/0"), ("dialogue", "o/dialogue_segments/0"))}}
        for name in ("motion", "dialogue"):
            av.director_resolution(resolution, chosen, name, chosen["o"], {"start_s": 0, "end_s": 5})
        with self.assertRaisesRegex(ValueError, "missing this check"):
            av.director_resolution(resolution, chosen, "audio", chosen["o"], {"start_s": 0, "end_s": 5})

    def test_real_blind_adapter_has_distinct_cache_and_cannot_inherit_qualification(self):
        from tests.video.test_gemini_observer import FakeTransport, lifecycle
        # Even an explicitly matching mock qualification must not promote this
        # evidence-only mode into a dynamic-check pass or accepted segment.
        policy = self.policy(qualify=True)["observation_policy"]
        qualifications = copy.deepcopy(policy["qualifications"])
        qualifications.append({**qualifications[0], "prompt_version": BLIND_PROMPT_VERSION})
        self.policy(qualifications=qualifications)
        transport = FakeTransport(*lifecycle(), *lifecycle())
        blind_observer = GeminiObserver("fake_api_key_for_tests", transport, mode="blind_facts", poll_interval=0)
        self.observer = blind_observer
        state = self.observe(observation_mode="blind_facts", reference_ids=[])
        record = state["av_observations"]["observation_0001"]
        request = record["request"]
        self.assertEqual(request["observation_mode"], "blind_facts")
        self.assertEqual(request["prompt_version"], BLIND_PROMPT_VERSION)
        self.assertEqual(request["system_prompt_sha256"], BLIND_SYSTEM_PROMPT_SHA256)
        self.assertEqual(request["spec"]["target"], "Synthetic test only")
        self.assertEqual(request["reference_hash"], runtime.review_context(state, "attempt_0001")[1]["reference_hash"])
        self.assertEqual(record["validated_capabilities"], [])
        self.assertEqual(record["qualifications"], [])
        self.assertEqual(record["reserved_media_seconds"], 5)
        self.assertEqual(record["result"]["local_request_binding"]["request_sha256"], record["request_hash"])
        self.assertEqual(len(transport.requests), 4)
        blind_body = json.loads(transport.requests[2].data)
        self.assertEqual(len(blind_body["contents"][0]["parts"]), 2)
        self.assertNotIn("Synthetic test only", json.dumps(blind_body))
        with self.assertRaisesRegex(ValueError, "qualification"):
            self.call("review", {"attempt_id": "attempt_0001", "report": self.report(state)})
        self.assertEqual(self.call("status")["attempts"][0]["status"], "review_pending")
        self.observer = GeminiObserver("fake_api_key_for_tests", transport, poll_interval=0)
        state = self.observe(observation_mode="comparison")
        self.assertEqual(len(state["av_observations"]), 2)
        self.assertEqual(state["av_observations"]["observation_0002"]["validated_capabilities"], ["motion"])
        self.assertNotEqual(record["request_hash"], state["av_observations"]["observation_0002"]["request_hash"])
        self.assertEqual(len(transport.requests), 8)
        self.observer = blind_observer
        again = self.observe(observation_mode="blind_facts", reference_ids=[])
        self.assertEqual(len(again["av_observations"]), 2)
        self.assertEqual(len(transport.requests), 8)

    def test_blind_runtime_rejects_reference_attachments_before_charge_or_http(self):
        from tests.video.test_gemini_observer import FakeTransport
        transport = FakeTransport()
        self.observer = GeminiObserver("fake_api_key_for_tests", transport, mode="blind_facts")
        with self.assertRaisesRegex(ValueError, "empty reference_ids"):
            self.observe(reference_ids=["ref"], observation_mode="blind_facts")
        self.assertEqual(transport.requests, [])
        self.assertEqual(self.call("status")["av_observations"], {})

    def test_runtime_rejects_result_with_changed_prompt_identity_and_keeps_charge(self):
        original_observe = self.observer.observe

        def wrong_identity(request, directory, callback):
            return {**original_observe(request, directory, callback), "prompt_version": "different-prompt"}

        self.observer.observe = wrong_identity
        with self.assertRaisesRegex(ValueError, "prompt identity changed"):
            self.observe()
        record = self.call("status")["av_observations"]["observation_0001"]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["reserved_media_seconds"], 5)
        self.assertEqual(record["cleanup_status"], "complete")

    def test_qualified_observation_accepts_and_final_requires_own_report(self):
        self.policy(qualify=True)
        state = self.observe()
        self.call("review", {"attempt_id": "attempt_0001", "report": self.report(state)})
        state = self.call("assemble")
        with self.assertRaisesRegex(ValueError, "another run or target"):
            self.call("final-review", self.report(state, target="final"))
        state = self.call("observe-av", {"target_id": "final", "qualification_scope": "fixture"}, live=True)
        state = self.call("final-review", self.report(state, target="final", oid="observation_0002"))
        self.assertEqual(state["final"]["status"], "accepted")
        self.assertFalse(state["observation_policy"]["active"])
        self.assertEqual(self.observer.calls, 2)

    def test_cache_avoids_charge_but_model_change_cannot_inherit_qualification(self):
        self.policy(qualify=True)
        state = self.observe()
        again = self.observe()
        self.assertEqual(self.observer.calls, 1)
        self.assertEqual(len(again["av_observations"]), 1)
        self.observer.model = "unexpected-version"
        self.call("inspect-window", {"target_id": "attempt_0001", "start_s": 1, "end_s": 2})
        state = self.observe(window_id="window_0001")
        self.assertEqual(state["av_observations"]["observation_0002"]["validated_capabilities"], [])

    def test_qualification_revocation_blocks_old_receipt_without_new_observation(self):
        self.policy(qualify=True)
        state = self.observe()
        self.policy()
        with self.assertRaisesRegex(ValueError, "qualification"):
            self.call("review", {"attempt_id": "attempt_0001", "report": self.report(state)})
        self.assertEqual(self.observer.calls, 1)

    def test_unknown_action_is_not_cleared_by_an_event_description(self):
        self.policy(qualify=True)
        self.observer.uncertainties = [{"start_s": 3, "end_s": 5, "description": "completion occluded", "evidence": "not visible"}]
        state = self.observe()
        with self.assertRaisesRegex(ValueError, "uncertainties/0"):
            self.call("review", {"attempt_id": "attempt_0001", "report": self.report(state)})

    def test_cleanup_recovers_file_id_durable_before_runtime_callback(self):
        self.observer.failure = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.observe()
        state = self.call("status")
        artifact = state["av_observations"]["observation_0001"]
        records = artifact["remote_files"]
        artifact["remote_files"] = []
        runtime.save(self.runs / "test/state.json", state)
        manifest = self.runs / "test/observation_0001/gemini-remote-files.json"
        manifest.write_text(json.dumps({"remote_files": records, "remote_upload_outcome_unknown": True}))
        state = self.call("cleanup-observation", {"observation_id": "observation_0001", "recovery_note": "synthetic interrupted callback"}, live=True)
        record = state["av_observations"]["observation_0001"]
        self.assertEqual(record["remote_files"][0]["cleanup_status"], "deleted")
        self.assertEqual(record["cleanup_status"], "cleanup_pending")

    def test_final_media_and_output_token_reservations_are_preserved(self):
        for options, message in (({"max_media_seconds": 9}, "media seconds reserved"),
                                 ({"max_total_output_tokens": 4096}, "output tokens reserved")):
            self.policy(**options)
            with self.assertRaisesRegex(ValueError, message):
                self.observe()
        self.assertEqual(self.observer.calls, 0)

    def test_audio_reference_budget_uses_actual_duration(self):
        audio = self.runs / "reference.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=5", str(audio)], check=True, timeout=30)
        state = self.call("status")
        ref = state["evidence"]["ref"]
        ref.update(observation_path=str(audio), observation_mime_type="audio/wav", observation_sha256=av.file_hash(audio), duration_s=0.01)
        runtime.save(self.runs / "test/state.json", state)
        self.policy(reference_transfer_ids=["ref"])
        with self.assertRaisesRegex(ValueError, "actual audio"):
            self.observe(reference_ids=["ref"])
        self.assertEqual(self.observer.calls, 0)
        ref["duration_s"] = 4.9
        state["observation_policy"] = self.call("status")["observation_policy"]
        runtime.save(self.runs / "test/state.json", state)
        state = self.observe(reference_ids=["ref"])
        self.assertEqual(state["av_observations"]["observation_0001"]["reserved_media_seconds"], 10.0)

    def test_failure_reserved_budget_and_uncertain_cleanup_preserved(self):
        self.policy(max_calls=2)
        self.observer.failure = GeminiObserverError("timeout", artifact={"remote_upload_outcome_unknown": True, "remote_files": [], "cleanup_status": "cleanup_pending"})
        with self.assertRaises(GeminiObserverError):
            self.observe()
        state = self.call("status")
        self.assertEqual(state["av_observations"]["observation_0001"]["reserved_media_seconds"], 5)
        state = self.call("cleanup-observation", {"observation_id": "observation_0001"}, live=True)
        self.assertEqual(state["av_observations"]["observation_0001"]["cleanup_status"], "cleanup_pending")
        self.observer.failure = None
        with self.assertRaisesRegex(ValueError, "reserved for final"):
            self.observe(retry_reason="bounded retry after failure")
        self.assertEqual(self.observer.calls, 1)

    def test_crash_persists_remote_file_and_requires_explicit_recovery(self):
        self.observer.failure = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.observe()
        state = self.call("status")
        self.assertEqual(state["av_observations"]["observation_0001"]["status"], "in_progress")
        self.assertEqual(len(state["av_observations"]["observation_0001"]["remote_files"]), 1)
        self.observer.failure = None
        with self.assertRaisesRegex(ValueError, "reconciled"):
            self.observe()
        self.call("cleanup-observation", {"observation_id": "observation_0001", "recovery_note": "Fake process interrupted; keep charge"}, live=True)
        self.observe(retry_reason="explicit bounded retry")
        self.assertEqual(self.observer.calls, 2)

    def test_missing_audio_hallucination_fails_and_report_is_saved(self):
        self.observer.extra_audio = True
        with self.assertRaisesRegex(ValueError, "without audio"):
            self.observe()
        record = self.call("status")["av_observations"]["observation_0001"]
        self.assertEqual(record["status"], "failed")
        self.assertTrue(Path(record["result"]["raw_response_path"]).exists())

    def test_local_only_evidence_cannot_pass_full_video(self):
        self.policy(qualify=True)
        state = self.call("inspect-window", {"target_id": "attempt_0001", "start_s": 1, "end_s": 2, "fps": 4})
        state = self.observe(window_id="window_0001")
        with self.assertRaisesRegex(ValueError, "full-video"):
            self.call("review", {"attempt_id": "attempt_0001", "report": self.report(state)})

    def test_current_failure_cannot_be_omitted_from_review(self):
        self.policy(qualify=True)
        self.observer.findings = [{"start_s": 1, "end_s": 2, "check": "motion", "status": "fail", "description": "stopped", "evidence": "same position"}]
        self.observe()
        self.observer.findings = []
        self.call("inspect-window", {"target_id": "attempt_0001", "start_s": 1, "end_s": 3})
        state = self.observe(window_id="window_0001")
        report = self.report(state)
        report["observation_ids"].append("observation_0002")
        report["checks"]["motion"]["evidence_refs"].append("observation_0002/events/0")
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.call("review", {"attempt_id": "attempt_0001", "report": report})
        report["checks"]["motion"]["resolutions"] = {"observation_0001/findings/0": "observation_0002/events/0"}
        self.call("review", {"attempt_id": "attempt_0001", "report": report})

    def test_human_confirmation_scoped_to_hash_and_check(self):
        state = self.call("status")
        sha = state["attempts"][0]["observation"]["media_sha256"]
        self.call("record-human-review", {"target_id": "attempt_0001", "media_sha256": sha, "checks": ["motion"],
                  "user_confirmation": "Mock user confirmation for tests", "observation_description": "entire synthetic video"})
        report = self.report()
        report["observation_ids"] = []
        report["checks"]["motion"]["human_review_id"] = "human_0001"
        state = self.call("review", {"attempt_id": "attempt_0001", "report": report})
        self.assertEqual(state["attempts"][0]["status"], "accepted")

    def test_stale_reference_spec_raw_response_and_media_are_rejected(self):
        self.policy(qualify=True)
        state = self.observe()
        report = self.report(state)
        original = copy.deepcopy(state)
        for mutate in (lambda s: s["plan"].update(script="different"),
                       lambda s: s["evidence"]["ref"].update(claim="different")):
            state = copy.deepcopy(original)
            mutate(state)
            runtime.save(self.runs / "test/state.json", state)
            with self.assertRaises(ValueError):
                self.call("review", {"attempt_id": "attempt_0001", "report": report})
        runtime.save(self.runs / "test/state.json", original)
        Path(original["av_observations"]["observation_0001"]["result"]["raw_response_path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "raw response changed"):
            self.call("review", {"attempt_id": "attempt_0001", "report": report})

    def test_legacy_accepted_checkpoint_blocks_next_shot_and_can_reopen_without_generation(self):
        state = self.call("status")
        state["plan"]["shots"].append({"id": "s2", "goal": "next", "duration_s": 5, "required_checks": ["motion"], "evidence_ids": ["ref"]})
        old = self.report(state)
        old.pop("observation_ids")
        old["playback_evidence"] = "Old legacy playback statement"
        state["attempts"][0].update(status="accepted", reviews=[old])
        state["accepted"]["s1"] = "attempt_0001"
        runtime.save(self.runs / "test/state.json", state)
        with self.assertRaises(ValueError):
            self.call("prepare", {"shot_id": "s2", "endpoint": "minimax/h3-max/text-to-video",
                                  "arguments": {"prompt": "test", "duration": 5, "prompt_expansion_mode": "balanced"}})
        state = self.call("reopen-review", {"target_id": "attempt_0001", "reason": "Legacy review lacks actual evidence"})
        self.assertEqual(state["attempts"][0]["status"], "review_pending")
        self.assertEqual(state["accepted"], {})
        self.assertEqual(self.observer.calls, 0)

    def test_real_adapter_contract_through_fake_http(self):
        from anigen.video.gemini_observer import GeminiObserver
        from tests.video.test_gemini_observer import FakeTransport, lifecycle
        transport = FakeTransport(*lifecycle())
        self.observer = GeminiObserver("fake_api_key_for_tests", transport, poll_interval=0)
        self.policy(qualify=True)
        state = self.observe()
        record = state["av_observations"]["observation_0001"]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["cleanup_status"], "complete")
        self.call("review", {"attempt_id": "attempt_0001", "report": self.report(state)})
        self.assertFalse(transport.steps)

    def test_two_shot_final_requires_seam_window_in_addition_to_full_observation(self):
        state = self.call("status")
        state["plan"]["shots"].append({"id": "s2", "goal": "second red clip", "duration_s": 5, "required_checks": ["motion"], "evidence_ids": ["ref"]})
        runtime.save(self.runs / "test/state.json", state)
        self.policy(qualify=True, imported_target_ids=["attempt_0001", "attempt_0002"])
        self.observe()
        self.call("review", {"attempt_id": "attempt_0001", "report": self.report()})
        self.call("import-video", {"shot_id": "s2", "media_path": str(self.video), "user_instruction": "synthetic second clip"})
        state = self.call("observe-av", {"target_id": "attempt_0002", "qualification_scope": "fixture"}, live=True)
        self.call("review", {"attempt_id": "attempt_0002", "report": self.report(state, target="attempt_0002", oid="observation_0002")})
        self.call("assemble")
        state = self.call("observe-av", {"target_id": "final", "qualification_scope": "fixture"}, live=True)
        report = self.report(state, target="final", oid="observation_0003")
        with self.assertRaisesRegex(ValueError, "seam inspected"):
            self.call("final-review", report)
        state = self.call("inspect-window", {"target_id": "final", "start_s": 4.5, "end_s": 5.5})
        window = state["inspection_windows"]["window_0001"]
        report["boundary_reviews"] = [{"boundary_s": state["final"]["boundaries_s"][0], "window_id": "window_0001",
                                       "observed_frames": [f["path"] for f in window["media"]["frames"]], "reason": "synthetic seam fixture"}]
        self.assertEqual(self.call("final-review", report)["final"]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
