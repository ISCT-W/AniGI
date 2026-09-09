import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from anigen.video.gemini_observer import (
    DEFAULT_MODEL, MAX_FILE_BYTES, MAX_OUTPUT_TOKENS, MAX_RESPONSE_BYTES,
    ORIGIN, PROMPT_VERSION, RESPONSE_SCHEMA, SYSTEM_PROMPT_SHA256,
    BLIND_PROMPT_VERSION, BLIND_RESPONSE_SCHEMA, BLIND_SYSTEM_PROMPT_SHA256,
    GeminiObserver, GeminiObserverError, _NoRedirect,
)


def observed():
    return {"events": [{"start_s": 0, "end_s": 1, "description": "Person raises an arm", "evidence": "Visible arm"}],
            "dialogue_segments": [], "audio_events": [], "findings": [], "uncertainties": []}


def envelope(response=None, **updates):
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(response or observed())}]}}],
            "modelVersion": DEFAULT_MODEL, "responseId": "response-123",
            "usageMetadata": {"totalTokenCount": 99}, **updates}


def remote(name="files/example-1", state="ACTIVE", mime="video/mp4"):
    return {"name": name, "uri": f"{ORIGIN}/v1beta/{name}", "mimeType": mime, "state": state}


class FakeResponse(io.BytesIO):
    def __init__(self, value=None, *, headers=None, status=200, url=None):
        data = value if isinstance(value, bytes) else json.dumps(value if value is not None else {}).encode()
        super().__init__(data)
        self.headers = headers or {}
        self.status = status
        self.url = url

    def geturl(self):
        return self.url


class FakeTransport:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        assert timeout > 0
        if not self.steps:
            raise AssertionError("Unexpected HTTP call")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            step = step(request)
        step.url = step.url or request.full_url
        return step


def upload_start(url=None):
    return FakeResponse(headers={"X-Goog-Upload-URL": url or f"{ORIGIN}/upload/v1beta/files?upload_id=opaque-token"})


def lifecycle(response=None, *, state="ACTIVE", cleanup=None):
    steps = [upload_start(), FakeResponse({"file": remote(state=state)})]
    if state == "PROCESSING":
        steps.append(FakeResponse(remote()))
    steps += [FakeResponse(response if response is not None else envelope()), cleanup or FakeResponse()]
    return steps


class GeminiObserverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.media = self.directory / "sample.mp4"
        self.media.write_bytes(b"offline-original-video-and-audio-stream-bytes")
        self.request = {"media_path": str(self.media), "uploaded_media_sha256": hashlib.sha256(self.media.read_bytes()).hexdigest(),
                        "requested_fps": 4, "spec": {"required_checks": ["action"], "action": "raise arm"}, "references": []}

    def tearDown(self):
        self.temporary.cleanup()

    def observer(self, transport):
        return GeminiObserver("fake_secret_key_for_tests", transport, poll_interval=0)

    def run_observe(self, transport, callback=None):
        return self.observer(transport).observe(self.request, self.directory / "result", callback)

    def test_lifecycle_persists_before_polling_and_cleans(self):
        transport = FakeTransport(*lifecycle(state="PROCESSING"))
        callbacks = []

        def callback(record):
            self.assertEqual(len(transport.requests), 2)
            manifest = json.loads((self.directory / "result/gemini-remote-files.json").read_text())
            self.assertEqual(manifest["remote_files"][0]["name"], record["name"])
            callbacks.append(record)

        result = self.run_observe(transport, callback)
        self.assertEqual(result["response"], observed())
        self.assertEqual(result["cleanup_status"], "complete")
        self.assertEqual(result["returned_model"], DEFAULT_MODEL)
        self.assertEqual(result["usage"], {"totalTokenCount": 99})
        self.assertIsNone(result["actual_sampling_points"])
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(transport.requests[1].data, self.media.read_bytes())
        self.assertNotIn("X-goog-api-key", dict(transport.requests[1].header_items()))
        self.assertEqual([r.method for r in transport.requests], ["POST", "POST", "GET", "POST", "DELETE"])
        body = json.loads(transport.requests[3].data)
        self.assertEqual(body["contents"][0]["parts"][1]["videoMetadata"], {"fps": 4})
        self.assertEqual(body["generationConfig"]["responseJsonSchema"], RESPONSE_SCHEMA)
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], MAX_OUTPUT_TOKENS)
        self.assertEqual(hashlib.sha256(Path(result["raw_response_path"]).read_bytes()).hexdigest(), result["raw_response_sha256"])
        self.assertFalse(transport.steps)

    def test_reference_labels_and_original_bytes(self):
        ref = self.directory / "reference.png"
        ref.write_bytes(b"image-reference-data")
        self.request["references"] = [{"id": "fixture-character-1", "path": str(ref), "mime_type": "image/png",
                                       "sha256": hashlib.sha256(ref.read_bytes()).hexdigest(), "label": "costume"}]
        transport = FakeTransport(upload_start(), FakeResponse({"file": remote()}), upload_start(),
                                  FakeResponse({"file": remote("files/reference-1", mime="image/png")}),
                                  FakeResponse(envelope()), FakeResponse(), FakeResponse())
        result = self.run_observe(transport)
        self.assertEqual(transport.requests[3].data, ref.read_bytes())
        parts = json.loads(transport.requests[4].data)["contents"][0]["parts"]
        self.assertIn("TARGET", parts[0]["text"])
        self.assertIn("REFERENCE fixture-character-1", parts[2]["text"])
        self.assertNotIn("videoMetadata", parts[3])
        self.assertEqual(len(result["remote_files"]), 2)

    def test_blind_facts_sends_only_target_with_neutral_label_and_pins_local_request(self):
        # The unused attachment need not even be readable: a blind pass must
        # neither read its bytes nor upload its descriptive name or labels.
        canaries = ["PRIVATE_EXPECTED_ACTION_AT_ZERO", "PRIVATE_DESIRED_SHOT_GOAL",
                    "PRIVATE_REFERENCE_LABEL", "PRIVATE_PREVIOUS_REVIEW", "PRIVATE_RUN_NAME"]
        self.request.update(
            spec={"script": canaries[0], "target": canaries[1]},
            references=[{"id": canaries[2], "label": canaries[2], "path": str(self.directory / "unreadable.png"),
                         "mime_type": "image/png", "sha256": "0" * 64}],
            history={"previous_review": canaries[3]}, run_id=canaries[4],
            observation_mode="blind_facts", prompt_version=BLIND_PROMPT_VERSION)
        original = json.loads(json.dumps(self.request))
        transport = FakeTransport(*lifecycle())
        observer = GeminiObserver("fake_secret_key_for_tests", transport, mode="blind_facts", poll_interval=0)

        def mutate_request(_):
            self.request["spec"]["script"] = "callback-injected-expected-answer"
            self.request["references"].append({"label": "callback-injected-reference"})
            self.request["requested_fps"] = 1

        result = observer.observe(self.request, self.directory / "blind", mutate_request)
        body = json.loads(transport.requests[2].data)
        parts = body["contents"][0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], {"text": "TARGET"})
        self.assertEqual(parts[1]["videoMetadata"], {"fps": 4})
        self.assertEqual(transport.requests[1].data, self.media.read_bytes())
        self.assertEqual(body["generationConfig"]["responseJsonSchema"], BLIND_RESPONSE_SCHEMA)
        self.assertEqual(body["generationConfig"]["candidateCount"], 1)
        wire = b"\n".join(request.data or b"" for request in transport.requests).decode()
        for private_text in [*canaries, "callback-injected-expected-answer", "callback-injected-reference", str(self.directory)]:
            self.assertNotIn(private_text, wire)
        prompt = json.loads(Path(result["prompt_path"]).read_bytes())
        self.assertNotIn("spec", prompt)
        self.assertEqual(len(prompt["inputs"]), 1)
        self.assertEqual(prompt["inputs"][0]["label"], "TARGET")
        self.assertEqual(result["prompt_version"], BLIND_PROMPT_VERSION)
        self.assertEqual(result["system_prompt_sha256"], BLIND_SYSTEM_PROMPT_SHA256)
        self.assertNotEqual(result["system_prompt_sha256"], SYSTEM_PROMPT_SHA256)
        self.assertEqual(hashlib.sha256(body["systemInstruction"]["parts"][0]["text"].encode()).hexdigest(),
                         BLIND_SYSTEM_PROMPT_SHA256)
        self.assertEqual(result["prompt_hash"], hashlib.sha256(Path(result["prompt_path"]).read_bytes()).hexdigest())
        encoded = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(result["local_request_binding"]["request_sha256"], hashlib.sha256(encoded(original)).hexdigest())
        self.assertEqual(result["local_request_binding"]["spec_sha256"], hashlib.sha256(encoded(original["spec"])).hexdigest())
        self.assertEqual(result["local_request_binding"]["references_sha256"], hashlib.sha256(encoded(original["references"])).hexdigest())
        self.assertEqual(result["evidence_role"], "blind_facts_only")
        self.assertFalse(result["comparison_performed"])
        self.assertFalse(result["qualification_eligible"])
        self.assertNotIn("validated_capabilities", result)
        self.assertNotIn("accepted", result)
        self.assertEqual(result["cleanup_status"], "complete")
        self.assertEqual([request.method for request in transport.requests], ["POST", "POST", "POST", "DELETE"])
        self.assertFalse(transport.steps)

    def test_blind_facts_rejects_model_comparison_findings_and_preserves_raw_evidence(self):
        response = observed()
        response["findings"] = [{"check": "action", "status": "observed", "start_s": 0, "end_s": 1,
                                 "description": "Matches the supposed task", "evidence": "events[0]"}]
        transport = FakeTransport(*lifecycle(envelope(response)))
        observer = GeminiObserver("fake_secret_key_for_tests", transport, mode="blind_facts", poll_interval=0)
        with self.assertRaises(GeminiObserverError) as caught:
            observer.observe(self.request, self.directory / "blind-findings")
        self.assertEqual(caught.exception.code, "blind_comparison_findings_forbidden")
        artifact = caught.exception.artifact
        self.assertTrue(Path(artifact["raw_response_path"]).is_file())
        self.assertEqual(artifact["cleanup_status"], "complete")
        self.assertFalse(artifact["qualification_eligible"])
        self.assertEqual(len(transport.requests), 4)

    def test_blind_and_comparison_mode_identity_mismatch_fails_before_http(self):
        for mode, request_mode, version in [("blind_facts", "comparison", PROMPT_VERSION),
                                            ("blind_facts", "blind_facts", PROMPT_VERSION),
                                            ("comparison", "blind_facts", BLIND_PROMPT_VERSION)]:
            transport = FakeTransport()
            observer = GeminiObserver("fake_secret_key_for_tests", transport, mode=mode)
            with self.subTest(mode=mode, request_mode=request_mode, version=version):
                with self.assertRaises(GeminiObserverError) as caught:
                    observer.observe({**self.request, "observation_mode": request_mode, "prompt_version": version},
                                     self.directory / "mismatch")
                self.assertEqual(caught.exception.code, "observation_mode_contract_mismatch")
                self.assertEqual(transport.requests, [])
        with self.assertRaises(ValueError):
            GeminiObserver("fake_secret_key_for_tests", FakeTransport(), mode="arbitrary")

    def test_directing_contract_binds_target_reference_video_and_actual_observations(self):
        reference = self.directory / "action-reference.mp4"
        reference.write_bytes(b"original-reference-action-not-the-target-choreography")
        ref_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        self.request["references"] = [{"id": "fixture-action", "path": str(reference),
                                       "mime_type": "video/mp4", "sha256": ref_hash,
                                       "label": "Sword arc mechanics; source excerpt begins at 90 seconds"}]
        self.request["spec"] = {"required_checks": ["action_timing"],
                                "script": "The actor starts a continuous attack at 0 seconds"}
        response = {"events": [
            {"start_s": 0, "end_s": 3.5, "description": "Actor holds raised blade while snow falls",
             "evidence": "Body, supporting foot and blade retain their pose; the camera moves closer"},
            {"start_s": 3.5, "end_s": 5, "description": "Actor rotates torso and swings the blade upward",
             "evidence": "Torso turns, arm extends and blade traverses the lower-to-upper diagonal"}],
            "dialogue_segments": [], "audio_events": [], "uncertainties": [],
            "findings": [{"check": "action_timing", "status": "fail", "start_s": 0, "end_s": 3.5,
                          "description": "Expected action from the opening; visible swing begins about 3.5 seconds",
                          "evidence": "events[0], events[1]; camera approach and snow alone are not actor action"}]}
        transport = FakeTransport(upload_start(), FakeResponse({"file": remote()}), upload_start(),
                                  FakeResponse({"file": remote("files/action-reference")}),
                                  FakeResponse(envelope(response)), FakeResponse(), FakeResponse())
        result = self.run_observe(transport)
        body = json.loads(transport.requests[4].data)
        parts = body["contents"][0]["parts"]
        contract_path = Path(result["prompt_path"])
        contract = json.loads(contract_path.read_bytes())
        self.assertEqual(result["prompt_version"], PROMPT_VERSION)
        self.assertEqual(result["prompt_hash"], hashlib.sha256(contract_path.read_bytes()).hexdigest())
        self.assertEqual(result["system_prompt_sha256"], SYSTEM_PROMPT_SHA256)
        self.assertEqual(hashlib.sha256(body["systemInstruction"]["parts"][0]["text"].encode()).hexdigest(),
                         SYSTEM_PROMPT_SHA256)
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], contract["system"])
        self.assertEqual(body["generationConfig"]["responseJsonSchema"], contract["response_schema"])
        self.assertEqual([item["role"] for item in contract["inputs"]], ["target", "reference"])
        self.assertEqual([item["sha256"] for item in contract["inputs"]],
                         [self.request["uploaded_media_sha256"], ref_hash])
        for index, item in enumerate(contract["inputs"]):
            self.assertEqual(parts[index * 2], {"text": item["label"]})
            self.assertEqual(parts[index * 2 + 1]["fileData"]["fileUri"], result["remote_files"][index]["uri"])
            self.assertEqual(parts[index * 2 + 1]["videoMetadata"], {"fps": 4})
            self.assertNotIn("path", item)
        self.assertEqual(transport.requests[3].data, reference.read_bytes())
        self.assertEqual(json.loads(parts[-1]["text"].split("\n", 1)[1]), self.request["spec"])
        # Actual target timing is preserved: neither a reference's source offset
        # nor the expected time in the plan is substituted into model evidence.
        self.assertEqual(result["response"], response)
        self.assertEqual(result["semantic_consistency"], "not_locally_validated")
        self.assertIsNone(contract["sampling"]["actual_sampling_points"])
        self.assertEqual([r.method for r in transport.requests].count("POST"), 5)

    def test_recorded_comparison_specification_is_frozen_before_upload_callbacks(self):
        original_spec = json.loads(json.dumps(self.request["spec"]))
        transport = FakeTransport(*lifecycle())

        def change_request_after_upload(record):
            self.request["spec"]["action"] = "different action after upload"
            self.request["requested_fps"] = 1
            self.request["max_output_tokens"] = 1

        result = self.run_observe(transport, change_request_after_upload)
        body = json.loads(transport.requests[2].data)
        parts = body["contents"][0]["parts"]
        contract = json.loads(Path(result["prompt_path"]).read_bytes())
        self.assertEqual(contract["spec"], original_spec)
        self.assertEqual(json.loads(parts[-1]["text"].split("\n", 1)[1]), original_spec)
        self.assertEqual(parts[1]["videoMetadata"], {"fps": 4})
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], MAX_OUTPUT_TOKENS)
        self.assertEqual(result["requested_fps"], 4)

    def test_contradictory_model_findings_are_preserved_without_semantic_certification(self):
        response = observed()
        response["events"][0].update(start_s=3.5, end_s=4)
        response["findings"] = [{"check": "action", "status": "observed", "start_s": 0, "end_s": 1,
                                 "description": "Arm raising starts at opening as specified",
                                 "evidence": "events[0]"}]
        result = self.run_observe(FakeTransport(*lifecycle(envelope(response))))
        # Shape validation is not a semantic detector. Preserve contradictions
        # for the supervising agent; do not rewrite them into fabricated success.
        self.assertEqual(result["response"], response)
        self.assertEqual(result["semantic_consistency"], "not_locally_validated")
        self.assertNotIn("accepted", result)
        self.assertNotIn("validated_capabilities", result)

    def test_prompt_audit_written_before_upload_and_partial_record_cannot_be_reused(self):
        def inspect_before_upload(request):
            contract = json.loads((self.directory / "result/gemini-prompt.json").read_text())
            self.assertEqual(contract["version"], PROMPT_VERSION)
            self.assertEqual(contract["inputs"][0]["sha256"], self.request["uploaded_media_sha256"])
            return upload_start()

        transport = FakeTransport(inspect_before_upload, FakeResponse({"file": remote()}),
                                  FakeResponse(envelope()), FakeResponse())
        self.run_observe(transport)
        interrupted = self.directory / "interrupted"
        interrupted.mkdir()
        (interrupted / "gemini-prompt.json").write_text("{}")
        retry_transport = FakeTransport()
        with self.assertRaises(GeminiObserverError) as caught:
            self.observer(retry_transport).observe(self.request, interrupted)
        self.assertEqual(caught.exception.code, "observation_directory_already_used")
        self.assertEqual(retry_transport.requests, [])

    def test_failed_processing_cleans_without_generation(self):
        transport = FakeTransport(upload_start(), FakeResponse({"file": remote(state="FAILED")}), FakeResponse())
        with self.assertRaises(GeminiObserverError) as caught:
            self.run_observe(transport)
        self.assertEqual(caught.exception.code, "file_processing_failed")
        self.assertEqual(caught.exception.artifact["cleanup_status"], "complete")
        self.assertFalse(caught.exception.artifact["generation_request_started"])
        self.assertEqual([r.method for r in transport.requests], ["POST", "POST", "DELETE"])

    def test_processing_poll_is_bounded(self):
        transport = FakeTransport(upload_start(), FakeResponse({"file": remote(state="PROCESSING")}), FakeResponse())
        with patch("anigen.video.gemini_observer.time.monotonic", side_effect=[0, 121]):
            with self.assertRaises(GeminiObserverError) as caught:
                self.run_observe(transport)
        self.assertEqual(caught.exception.code, "file_processing_timeout")
        self.assertEqual(len(transport.requests), 3)

    def test_malformed_and_empty_responses_persist_raw_and_cleanup(self):
        cases = [b"not JSON", {"candidates": []}, envelope(candidates=[{"finishReason": "MAX_TOKENS"}]),
                 envelope(candidates=[{"finishReason": "STOP", "content": {"parts": []}}]),
                 envelope(candidates=[{"finishReason": "STOP", "content": {"parts": [{"text": "{}"}]}}]),
                 envelope(candidates=[{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps({key: [] for key in observed()})}]}}])]
        for index, response in enumerate(cases):
            with self.subTest(index=index):
                transport = FakeTransport(*lifecycle(response))
                with self.assertRaises(GeminiObserverError) as caught:
                    self.observer(transport).observe(self.request, self.directory / f"invalid-{index}")
                self.assertTrue(Path(caught.exception.artifact["raw_response_path"]).is_file())
                self.assertEqual(caught.exception.artifact["cleanup_status"], "complete")
                self.assertEqual(transport.requests[-1].method, "DELETE")

    def test_no_automatic_retry_after_ambiguous_generation_timeout(self):
        transport = FakeTransport(upload_start(), FakeResponse({"file": remote()}),
                                  URLError("fake_secret_key_for_tests potentially exposed by transport"), FakeResponse())
        with self.assertRaises(GeminiObserverError) as caught:
            self.run_observe(transport)
        self.assertNotIn("fake_secret", str(caught.exception))
        self.assertTrue(caught.exception.artifact["generation_request_started"])
        self.assertEqual(len(transport.requests), 4)
        self.assertEqual(caught.exception.artifact["cleanup_status"], "complete")

    def test_lost_upload_receipt_is_unknown_even_with_empty_owned_files(self):
        transport = FakeTransport(upload_start(), TimeoutError("sensitive upload token"))
        with self.assertRaises(GeminiObserverError) as caught:
            self.run_observe(transport)
        self.assertEqual(caught.exception.artifact["remote_files"], [])
        self.assertTrue(caught.exception.artifact["remote_upload_outcome_unknown"])
        self.assertEqual(caught.exception.artifact["cleanup_status"], "cleanup_pending")
        manifest = json.loads((self.directory / "result/gemini-remote-files.json").read_text())
        self.assertTrue(manifest["remote_upload_outcome_unknown"])
        self.assertEqual(len(transport.requests), 2)
        self.assertNotIn("sensitive", str(caught.exception))

    def test_finalize_interruption_writes_unknown_before_request_and_preserves_it(self):
        seen_before_request = []

        def interrupted_finalize(request):
            manifest = json.loads((self.directory / "result/gemini-remote-files.json").read_text())
            seen_before_request.append(manifest)
            raise KeyboardInterrupt()

        transport = FakeTransport(upload_start(), interrupted_finalize)
        with self.assertRaises(KeyboardInterrupt):
            self.run_observe(transport)
        self.assertTrue(seen_before_request[0]["remote_upload_outcome_unknown"])
        manifest = json.loads((self.directory / "result/gemini-remote-files.json").read_text())
        self.assertTrue(manifest["remote_upload_outcome_unknown"])
        self.assertEqual(manifest["cleanup_status"], "cleanup_pending")
        self.assertEqual(manifest["remote_files"], [])
        self.assertEqual(len(transport.requests), 2)

    def test_second_upload_interruption_cleans_known_file_but_keeps_unknown(self):
        reference = self.directory / "reference.png"
        reference.write_bytes(b"reference-image")
        self.request["references"] = [{"id": "reference", "path": str(reference), "mime_type": "image/png",
                                       "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}]
        before_second_request = []

        def interrupted_finalize(request):
            before_second_request.append(json.loads((self.directory / "result/gemini-remote-files.json").read_text()))
            raise KeyboardInterrupt()

        transport = FakeTransport(upload_start(), FakeResponse({"file": remote()}), upload_start(),
                                  interrupted_finalize, FakeResponse())
        with self.assertRaises(KeyboardInterrupt):
            self.run_observe(transport)
        self.assertTrue(before_second_request[0]["remote_upload_outcome_unknown"])
        self.assertEqual(before_second_request[0]["remote_files"][0]["name"], "files/example-1")
        manifest = json.loads((self.directory / "result/gemini-remote-files.json").read_text())
        self.assertTrue(manifest["remote_upload_outcome_unknown"])
        self.assertEqual(manifest["cleanup_status"], "cleanup_pending")
        self.assertEqual(manifest["remote_files"][0]["cleanup_status"], "deleted")
        self.assertEqual([item.method for item in transport.requests], ["POST", "POST", "POST", "POST", "DELETE"])

    def test_callback_failure_cleans_and_does_not_expose_exception(self):
        transport = FakeTransport(upload_start(), FakeResponse({"file": remote()}), FakeResponse())

        def failed_callback(record):
            raise RuntimeError("secret callback contents")

        with self.assertRaises(GeminiObserverError) as caught:
            self.run_observe(transport, failed_callback)
        self.assertEqual(caught.exception.artifact["cleanup_status"], "complete")
        self.assertNotIn("secret", str(caught.exception))
        self.assertFalse(caught.exception.artifact["generation_request_started"])

    def test_cleanup_failure_is_recoverable_without_analysis(self):
        transport = FakeTransport(*lifecycle(cleanup=FakeResponse(status=503)))
        result = self.run_observe(transport)
        self.assertEqual(result["cleanup_status"], "cleanup_pending")
        cleanup_transport = FakeTransport(FakeResponse())
        recovered = self.observer(cleanup_transport).cleanup(result["remote_files"])
        self.assertEqual(recovered["cleanup_status"], "complete")
        self.assertEqual([r.method for r in cleanup_transport.requests], ["DELETE"])

    def test_cleanup_rejects_unowned_or_mismatched_or_traversal_records_before_network(self):
        transport = FakeTransport()
        good = {"name": "files/example-1", "uri": f"{ORIGIN}/v1beta/files/example-1", "owned": True}
        cases = [{**good, "owned": False}, {**good, "name": "files/../private"},
                 {**good, "uri": f"{ORIGIN}/v1beta/files/other"}, {**good, "uri": "https://evil.example/file"}]
        for record in cases:
            with self.subTest(record=record), self.assertRaises(GeminiObserverError):
                self.observer(transport).cleanup([record])
        self.assertEqual(transport.requests, [])

    def test_missing_deleted_file_is_successful_cleanup(self):
        transport = FakeTransport(FakeResponse(status=404))
        result = self.observer(transport).cleanup([{"name": "files/example-1", "uri": f"{ORIGIN}/v1beta/files/example-1", "owned": True}])
        self.assertEqual(result["cleanup_status"], "complete")

    def test_redirects_cannot_receive_key_or_media(self):
        urls = ["https://evil.example/upload", "http://generativelanguage.googleapis.com/upload/v1beta/files",
                f"{ORIGIN}:443/upload/v1beta/files", f"{ORIGIN}/upload/v1beta/files#fragment",
                ("https://" + "fixture:" + "fixture" + chr(64) + 'evil.example/upload/v1beta/files'),
                f"{ORIGIN}/other/path"]
        for index, url in enumerate(urls):
            transport = FakeTransport(upload_start(url))
            with self.subTest(url=url), self.assertRaises(GeminiObserverError):
                self.observer(transport).observe(self.request, self.directory / f"redirect-{index}")
            self.assertEqual(len(transport.requests), 1)
        self.assertIsNone(_NoRedirect().redirect_request(Request(ORIGIN), None, 302, "", {}, "https://evil.example"))

    def test_redirect_http_error_is_sanitized(self):
        transport = FakeTransport(HTTPError(ORIGIN, 302, "secret response", {"Location": "https://evil.example"}, None))
        with self.assertRaises(GeminiObserverError) as caught:
            self.run_observe(transport)
        self.assertEqual(caught.exception.code, "redirect_forbidden")
        self.assertNotIn("secret", str(caught.exception))

    def test_invalid_input_fails_before_any_network(self):
        cases = [{"uploaded_media_sha256": "bad"}, {"requested_model": "arbitrary-model"},
                 {"requested_fps": float("nan")}, {"requested_fps": True},
                 {"max_output_tokens": MAX_OUTPUT_TOKENS + 1},
                 {"references": [{"id": "ref", "path": str(self.media), "mime_type": "video/mp4", "sha256": "bad"}]}]
        for updates in cases:
            transport = FakeTransport()
            with self.subTest(updates=updates), self.assertRaises(GeminiObserverError):
                self.observer(transport).observe({**self.request, **updates}, self.directory / "invalid")
            self.assertEqual(transport.requests, [])

    def test_file_cap_and_response_cap(self):
        large = self.directory / "large.mp4"
        with large.open("wb") as stream:
            stream.truncate(MAX_FILE_BYTES + 1)
        with self.assertRaises(GeminiObserverError):
            self.observer(FakeTransport()).observe({**self.request, "media_path": str(large)}, self.directory / "large-result")
        transport = FakeTransport(upload_start(), FakeResponse({"file": remote()}),
                                  FakeResponse(b"x" * (MAX_RESPONSE_BYTES + 1)), FakeResponse())
        with self.assertRaises(GeminiObserverError) as caught:
            self.run_observe(transport)
        self.assertEqual(caught.exception.code, "response_too_large")
        self.assertEqual(caught.exception.artifact["cleanup_status"], "complete")

    def test_missing_and_malformed_key_do_not_use_network(self):
        for value in ["", "short", "spaces not allowed", "invalid\nheader", None]:
            transport = FakeTransport()
            with patch.dict("os.environ", {}, clear=True):
                with self.subTest(value=value), self.assertRaises(GeminiObserverError):
                    GeminiObserver(value, transport).preflight()
            self.assertEqual(transport.requests, [])

    def test_reusing_observation_directory_does_not_resubmit(self):
        self.run_observe(FakeTransport(*lifecycle()))
        transport = FakeTransport()
        with self.assertRaises(GeminiObserverError) as caught:
            self.run_observe(transport)
        self.assertEqual(caught.exception.code, "observation_directory_already_used")
        self.assertEqual(transport.requests, [])


if __name__ == "__main__":
    unittest.main()
