"""Offline contract and failure-path tests. No credentials or network required."""

import io
import json
import tempfile
import unittest
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from anigen.video.provider import FalProvider, ProviderError, SubmissionUncertain, request_duration, validate_request

T2V = "minimax/h3-max/text-to-video"
I2V = "minimax/h3-max/image-to-video"
R2V = "minimax/h3-max/reference-to-video"
ARGS = {"prompt": "A cat walking", "prompt_expansion_mode": "balanced"}
ROOT = "https://queue.fal.run/minimax/h3-max/requests/job-1"
RECEIPT = {"request_id": "job-1", "status_url": ROOT + "/status", "response_url": ROOT, "cancel_url": ROOT + "/cancel"}
MEDIA = "https://v3b.fal.media/files/video.mp4"


class Response(io.BytesIO):
    def __init__(self, body, url, *, status=200, headers=None):
        super().__init__(body if isinstance(body, bytes) else json.dumps(body).encode())
        self.url, self.status, self.headers = url, status, headers or {}

    def geturl(self):
        return self.url


class Transport:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, request, *, timeout):
        self.calls.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Response(reply, request.full_url)


class RequestValidationTests(unittest.TestCase):
    def test_supported_requests_and_duration(self):
        validate_request(T2V, ARGS)
        validate_request(I2V, {**ARGS, "end_image_url": "https://assets.example/end.png"})
        validate_request(R2V, {**ARGS, "reference_image_urls": ["https://assets.example/character.png"], "aspect_ratio": "adaptive"})
        self.assertEqual(request_duration(ARGS), 5)
        self.assertEqual(request_duration({"duration": 15}), 15)

    def test_reject_bad_types_and_unsupported_arguments(self):
        for extra in ({"duration": True}, {"duration": 5.1}, {"duration": 16}, {"duration": 4},
                      {"resolution": "2K"}, {"seed": False}, {"sync_mode": True},
                      {"prompt": " "}, {"prompt_expansion_mode": "unknown"}, {"image_url": MEDIA}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_request(T2V, {**ARGS, **extra})
        with self.assertRaises(ValueError):
            validate_request("minimax/h3-max/director", ARGS)
        with self.assertRaises(ValueError):
            validate_request(I2V, {**ARGS, "aspect_ratio": "16:9"})

    def test_reference_limits_and_audio_only(self):
        for refs in ({"reference_audio_urls": [MEDIA]}, {"reference_image_urls": [MEDIA] * 10},
                     {"reference_image_urls": [MEDIA] * 9, "reference_video_urls": [MEDIA] * 3, "reference_audio_urls": [MEDIA]},
                     {"reference_video_urls": [MEDIA] * 4}, {"reference_image_urls": MEDIA}):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                validate_request(R2V, {**ARGS, **refs})

    def test_reference_urls_reject_credentials_controls_and_non_https(self):
        for url in ("file:///tmp/private.png", "data:image/png;base64,abc", ("https://" + "fixture:" + "fixture" + chr(64) + 'a.example/a'), "https://a.example\n/path"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_request(I2V, {**ARGS, "image_url": url})


@patch.dict("os.environ", {"FAL_KEY": "unit-test-secret"}, clear=False)
class QueueTests(unittest.TestCase):
    def test_queue_roundtrip_uses_receipt_urls_and_no_retries(self):
        transport = Transport(RECEIPT, {"status": "COMPLETED", "request_id": "job-1"}, {"video": {"url": MEDIA}}, {"status": "CANCELLATION_REQUESTED"})
        provider = FalProvider(transport=transport)
        receipt = provider.submit(T2V, ARGS)
        self.assertEqual(provider.status(receipt)["status"], "COMPLETED")
        self.assertEqual(provider.result(receipt)["video"]["url"], MEDIA)
        self.assertEqual(provider.cancel(receipt)["status"], "CANCELLATION_REQUESTED")
        self.assertEqual([r.get_method() for r in transport.calls], ["POST", "GET", "GET", "PUT"])
        self.assertEqual(transport.calls[2].full_url, ROOT)
        self.assertEqual(json.loads(transport.calls[0].data), ARGS)
        self.assertEqual(transport.calls[0].get_header("Authorization"), "Key unit-test-secret")

    def test_no_key_no_network(self):
        transport = Transport()
        with patch.dict("os.environ", {"FAL_KEY": ""}), self.assertRaises(ProviderError):
            FalProvider(transport=transport).submit(T2V, ARGS)
        self.assertFalse(transport.calls)

    def test_no_credential_to_forged_or_mismatched_receipts(self):
        for url in ("https://evil.example/path", "http://queue.fal.run/minimax/h3-max/requests/job-1/status",
                    ROOT.replace("job-1", "job-2") + "/status", ROOT + "/cancel",
                    ("https://" + "fixture:" + "fixture" + chr(64) + 'evil.example/a'), ROOT + "/status?token=secret"):
            transport = Transport()
            with self.subTest(url=url), self.assertRaises(ValueError):
                FalProvider(transport=transport).status({**RECEIPT, "status_url": url})
            self.assertFalse(transport.calls)

    def test_ambiguous_submit_is_not_retried_and_error_is_sanitized(self):
        for failure in (URLError("unit-test-secret and private URL"), HTTPError("secret-url", 503, "secret", {}, None), IncompleteRead(b"unit-test-secret")):
            transport = Transport(failure)
            with self.assertRaises(SubmissionUncertain) as context:
                FalProvider(transport=transport).submit(T2V, ARGS)
            self.assertNotIn("unit-test-secret", str(context.exception))
            self.assertEqual(len(transport.calls), 1)

    def test_malformed_receipt_and_json_are_uncertain(self):
        for reply in (b"not json", {"request_id": "job-1"}, {**RECEIPT, "status_url": "https://evil.example/a"}):
            with self.subTest(reply=reply), self.assertRaises(SubmissionUncertain):
                FalProvider(transport=Transport(reply)).submit(T2V, ARGS)

    def test_queue_redirect_is_not_followed(self):
        transport = Transport(HTTPError(ROOT, 302, "moved", {"Location": "https://evil.example/"}, None))
        with self.assertRaises(ProviderError):
            FalProvider(transport=transport).status(RECEIPT)
        self.assertEqual(len(transport.calls), 1)

    def test_completed_with_error_remains_visible_to_orchestrator(self):
        result = FalProvider(transport=Transport({"status": "COMPLETED", "error": "model failed"})).status(RECEIPT)
        self.assertEqual(result["error"], "model failed")


class DownloadTests(unittest.TestCase):
    def test_download_without_authorization(self):
        transport = Transport(b"video bytes")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            self.assertEqual(FalProvider(transport=transport).download(MEDIA, path), path)
            self.assertEqual(path.read_bytes(), b"video bytes")
        self.assertIsNone(transport.calls[0].get_header("Authorization"))

    def test_redirect_checked_before_following(self):
        for location, allowed in (("https://v4.fal.media/files/clip.mp4", True), ("https://evil.example/a", False)):
            transport = Transport(HTTPError(MEDIA, 302, "moved", {"Location": location}, None), b"video")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "video.mp4"
                if allowed:
                    FalProvider(transport=transport).download(MEDIA, path)
                    self.assertEqual(len(transport.calls), 2)
                    self.assertTrue(all(r.get_header("Authorization") is None for r in transport.calls))
                else:
                    with self.assertRaises(ValueError):
                        FalProvider(transport=transport).download(MEDIA, path)
                    self.assertEqual(len(transport.calls), 1)
                    self.assertFalse(path.exists())

    def test_partial_oversized_empty_media_are_not_published(self):
        replies = [(b"12345", {}, 4), (b"123", {"Content-Length": "4"}, 10), (b"", {}, 10)]
        for body, headers, limit in replies:
            def transport(request, *, timeout):
                return Response(body, request.full_url, headers=headers)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "video.mp4"
                with self.assertRaises(ProviderError):
                    FalProvider(transport=transport, max_download_bytes=limit).download(MEDIA, path)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_files_are_never_overwritten(self):
        transport = Transport()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                FalProvider(transport=transport).download(MEDIA, path)
            self.assertEqual(path.read_bytes(), b"keep")
            self.assertFalse(transport.calls)


if __name__ == "__main__":
    unittest.main()
