import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from anigen.video.fal_upload import (
    CDN_ORIGIN, MAX_RESPONSE_BYTES, MAX_UPLOAD_BYTES, TOKEN_URL, UPLOAD_URL,
    FalMediaUploader, FalUploadError, FalUploadUncertain, _NoRedirect,
)


class FakeResponse(io.BytesIO):
    def __init__(self, value, *, url=None, status=200):
        super().__init__(value if isinstance(value, bytes) else json.dumps(value).encode())
        self.url, self.status, self.headers = url, status, {}

    def geturl(self):
        return self.url


class FakeTransport:
    def __init__(self, *steps):
        self.steps, self.requests = list(steps), []

    def __call__(self, request, timeout):
        self.requests.append(request)
        assert 0 < timeout <= 60
        if not self.steps:
            raise AssertionError("Unexpected request")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            step = step(request)
        step.url = step.url or request.full_url
        return step


def token(**updates):
    return FakeResponse({"token": "temporary-cdn-token", "token_type": "Bearer",
                         "base_url": CDN_ORIGIN, "expires_at": "2099-01-01T00:00:00+00:00", **updates})


def uploaded():
    return FakeResponse({"access_url": "https://v3.fal.media/files/example/frame.png"})


class FalMediaUploaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "private-name.png"
        self.data = b"\x89PNG\r\n\x1a\nreviewed-frame-bytes"
        self.path.write_bytes(self.data)
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.env = patch.dict(os.environ, {"FAL_KEY": "private-fal-key"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def run_upload(self, transport, **kwargs):
        return FalMediaUploader(transport=transport, **kwargs).upload(self.path, "image/png", self.digest)

    def test_success_keeps_key_on_rest_and_uploads_exact_hash_bytes(self):
        transport = FakeTransport(token(), uploaded())
        result = self.run_upload(transport)
        self.assertEqual([r.full_url for r in transport.requests], [TOKEN_URL, UPLOAD_URL])
        self.assertEqual([r.method for r in transport.requests], ["POST", "POST"])
        auth_request, media_request = transport.requests
        self.assertEqual(auth_request.get_header("Authorization"), "Key private-fal-key")
        self.assertEqual(auth_request.data, b"{}")
        self.assertEqual(media_request.get_header("Authorization"), "Bearer temporary-cdn-token")
        self.assertNotIn("private-fal-key", str(media_request.header_items()))
        self.assertEqual(media_request.data, self.data)
        self.assertEqual(media_request.get_header("Content-type"), "image/png")
        self.assertNotIn("private-name", media_request.get_header("X-fal-file-name"))
        self.assertEqual(result["source_sha256"], self.digest)
        self.assertEqual(result["source_path"], str(self.path.resolve()))
        self.assertEqual(result["size_bytes"], len(self.data))
        self.assertNotIn("temporary-cdn-token", json.dumps(result))
        self.assertNotIn("private-fal-key", json.dumps(result))

    def test_preflight_checks_only_key_without_http(self):
        transport = FakeTransport()
        uploader = FalMediaUploader(transport=transport)
        self.assertIsNone(uploader.preflight())
        for value in ("", "header\ninjection", "white space", "非ASCII"):
            with self.subTest(value=value), patch.dict(os.environ, {"FAL_KEY": value}):
                with self.assertRaises(FalUploadError) as caught:
                    uploader.preflight()
                self.assertFalse(caught.exception.outcome_unknown)
        self.assertEqual(transport.requests, [])

    def test_live_v3b_token_routes_bytes_only_to_exact_authenticated_cdn(self):
        # 2026-09-08 token-only live response selected this official v3 shard.
        # The SDK uses the authenticated base_url, not a hardcoded v3 hostname.
        for base in ("https://v3b.fal.media", "https://v3b.fal.media/"):
            with self.subTest(base=base):
                transport = FakeTransport(token(base_url=base), FakeResponse({
                    "access_url": "https://v3b.fal.media/files/example/frame.png"}))
                result = self.run_upload(transport)
                self.assertEqual([r.full_url for r in transport.requests], [
                    TOKEN_URL, "https://v3b.fal.media/files/upload"])
                self.assertEqual(transport.requests[1].data, self.data)
                self.assertEqual(transport.requests[1].get_header("Authorization"),
                                 "Bearer temporary-cdn-token")
                self.assertNotIn("private-fal-key", str(transport.requests[1].header_items()))
                self.assertEqual(result["access_url"], "https://v3b.fal.media/files/example/frame.png")

    def test_changed_hash_and_invalid_mime_never_contact_service(self):
        transport = FakeTransport()
        self.path.write_bytes(self.data + b"changed")
        with self.assertRaises(FalUploadError) as caught:
            self.run_upload(transport)
        self.assertEqual(caught.exception.code, "source_hash_mismatch")
        self.assertFalse(caught.exception.outcome_unknown)
        for mime in ("audio/wav", "text/plain", "image/png\r\nInjected: yes"):
            with self.subTest(mime=mime), self.assertRaises(FalUploadError):
                FalMediaUploader(transport=transport).upload(self.path, mime, self.digest)
        self.assertEqual(transport.requests, [])

    def test_signature_size_and_missing_file_fail_before_network(self):
        transport = FakeTransport()
        with self.assertRaises(FalUploadError):
            self.run_upload(transport, max_upload_bytes=8)
        self.path.write_bytes(b"plain text pretending to be png")
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaises(FalUploadError) as caught:
            FalMediaUploader(transport=transport).upload(self.path, "image/png", digest)
        self.assertEqual(caught.exception.code, "source_mime_mismatch")
        self.path.unlink()
        with self.assertRaises(FalUploadError):
            self.run_upload(transport)
        self.assertEqual(transport.requests, [])

    def test_mp4_has_explicit_mime_and_retains_original_bytes(self):
        payload = b"\x00\x00\x00\x18ftypisomvideo-tail"
        self.path.write_bytes(payload)
        transport = FakeTransport(token(base_url=CDN_ORIGIN + "/"), uploaded())
        result = FalMediaUploader(transport=transport).upload(
            self.path, "video/mp4", hashlib.sha256(payload).hexdigest())
        self.assertEqual(transport.requests[1].data, payload)
        self.assertEqual(result["mime_type"], "video/mp4")
        self.assertTrue(transport.requests[1].get_header("X-fal-file-name").endswith(".mp4"))

    def test_untrusted_token_host_cannot_receive_media(self):
        for base in ("https://v3.fal.media.evil.test", "https://evil.test", "https://v3.fal.media:443",
                     "https://v3.fal.media/files", "http://v3.fal.media", ("https://" + "fixture:" + "fixture" + chr(64) + 'v3.fal.media'),
                     "https://v3b.fal.media.evil.test", "https://v3b.fal.media:443",
                     "https://v3b.fal.media/files", "https://v3b.fal.media?unexpected=1",
                     "https://v3b.fal.media#fragment", ("https://" + "fixture:" + "fixture" + chr(64) + 'v3b.fal.media'),
                     "https://unverified.fal.media", "http://v3b.fal.media"):
            with self.subTest(base=base):
                transport = FakeTransport(token(base_url=base))
                with self.assertRaises(FalUploadError) as caught:
                    self.run_upload(transport)
                self.assertFalse(caught.exception.outcome_unknown)
                self.assertEqual(len(transport.requests), 1)

    def test_untrusted_token_fields_cannot_reuse_key_as_cdn_token(self):
        for fields in ({"token": "private-fal-key"}, {"token": "token\r\ninjection"},
                       {"token_type": "Bearer\r\ninjection"}, {"expires_at": "2000-01-01T00:00:00Z"},
                       {"expires_at": "2099-01-01T00:00:00"}):
            with self.subTest(fields=fields):
                transport = FakeTransport(token(**fields))
                with self.assertRaises(FalUploadError):
                    self.run_upload(transport)
                self.assertEqual(len(transport.requests), 1)

    def test_auth_failure_is_definite_sanitized_and_not_retried(self):
        error = HTTPError(TOKEN_URL, 401, "private-fal-key", {}, io.BytesIO(b"private-fal-key"))
        transport = FakeTransport(error)
        with self.assertRaises(FalUploadError) as caught:
            self.run_upload(transport)
        self.assertEqual(caught.exception.http_status, 401)
        self.assertEqual(caught.exception.stage, "token")
        self.assertFalse(caught.exception.outcome_unknown)
        self.assertNotIn("private-fal-key", str(caught.exception))
        self.assertEqual(len(transport.requests), 1)

    def test_upload_timeout_and_server_errors_are_unknown_without_retry(self):
        failures = [URLError("private-fal-key temporary-cdn-token"), TimeoutError("temporary-cdn-token"),
                    HTTPError(UPLOAD_URL, 500, "temporary-cdn-token", {}, None),
                    HTTPError(UPLOAD_URL, 408, "temporary-cdn-token", {}, None)]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                transport = FakeTransport(token(), failure)
                with self.assertRaises(FalUploadUncertain) as caught:
                    self.run_upload(transport)
                self.assertTrue(caught.exception.outcome_unknown)
                self.assertEqual(caught.exception.artifact["source_sha256"], self.digest)
                self.assertNotIn("temporary-cdn-token", str(caught.exception))
                self.assertEqual(len(transport.requests), 2)

    def test_redirect_handler_and_redirected_responses_are_rejected(self):
        self.assertIsNone(_NoRedirect().redirect_request(Request(UPLOAD_URL), None, 307, "redirect", {}, "https://evil.test"))
        for response in (FakeResponse({}, url="https://evil.test"), FakeResponse({}, status=307)):
            with self.subTest(response=response):
                transport = FakeTransport(token(), response)
                with self.assertRaises(FalUploadUncertain):
                    self.run_upload(transport)
                self.assertEqual(len(transport.requests), 2)
        transport = FakeTransport(FakeResponse({}, url="https://evil.test"))
        with self.assertRaises(FalUploadError) as caught:
            self.run_upload(transport)
        self.assertFalse(caught.exception.outcome_unknown)
        self.assertEqual(len(transport.requests), 1)

    def test_success_response_with_missing_or_untrusted_url_is_unknown(self):
        for value in ({}, {"access_url": "https://fal.media.evil.test/files/a.png"},
                      {"access_url": ("https://" + "fixture:" + "fixture" + chr(64) + 'v3.fal.media/files/a.png')},
                      {"access_url": "https://v3.fal.media/files/a.png?key=private-fal-key"},
                      {"access_url": "https://v3.fal.media/files/temporary-cdn-token.png"},
                      [], b"not-json", b"{" + b" " * MAX_RESPONSE_BYTES):
            with self.subTest(value=str(value)[:40]):
                transport = FakeTransport(token(), FakeResponse(value))
                with self.assertRaises(FalUploadUncertain):
                    self.run_upload(transport)
                self.assertEqual(len(transport.requests), 2)

    def test_upload_definite_client_rejection_does_not_consume_retry(self):
        transport = FakeTransport(token(), FakeResponse({}, status=413))
        with self.assertRaises(FalUploadError) as caught:
            self.run_upload(transport)
        self.assertFalse(caught.exception.outcome_unknown)
        self.assertEqual(caught.exception.http_status, 413)
        self.assertEqual(len(transport.requests), 2)

    def test_source_mutation_after_token_request_does_not_change_upload(self):
        def auth_response(_):
            self.path.write_bytes(b"later-local-change")
            return token()
        transport = FakeTransport(auth_response, uploaded())
        result = self.run_upload(transport)
        self.assertEqual(transport.requests[1].data, self.data)
        self.assertEqual(result["source_sha256"], self.digest)

    def test_config_cannot_exceed_hard_cap(self):
        for settings in ({"max_upload_bytes": MAX_UPLOAD_BYTES + 1}, {"max_upload_bytes": True},
                         {"timeout": 0}, {"timeout": 61}, {"timeout": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                FalMediaUploader(**settings)


if __name__ == "__main__":
    unittest.main()
