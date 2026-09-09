"""Bounded fal CDN v3 upload; authorization, budget and recovery belong to runtime.

Contract checked against fal-ai/fal's Python client on 2026-09-08. Only small
files are supported: token POST to rest.fal.ai, then one bytes POST to the
authenticated, explicitly allowed v3 CDN origin.
No dotenv loading, SDK, redirects, retries, fallback repository or token storage.
The injectable transport has urllib's open(Request, timeout=...) interface.
"""

from __future__ import annotations

import hashlib
import json
import os
from .settings import get_setting
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

TOKEN_URL = "https://rest.fal.ai/storage/auth/token?storage_type=fal-cdn-v3"
CDN_ORIGIN = "https://v3.fal.media"
UPLOAD_URL = CDN_ORIGIN + "/files/upload"
# The official token service also selects v3b (verified token-only 2026-09-08).
# Keep exact origins rather than trusting an arbitrary URL returned in JSON.
CDN_ORIGINS = frozenset((CDN_ORIGIN, "https://v3b.fal.media"))
_TOKEN_BASE_URLS = {base: origin for origin in CDN_ORIGINS
                    for base in (origin, origin + "/")}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "video/mp4": ".mp4"}


class FalUploadError(RuntimeError):
    """Sanitized failure, including whether media creation may have succeeded."""

    def __init__(self, code: str, *, stage: str = "preflight", http_status: int | None = None,
                 outcome_unknown: bool = False, artifact: dict | None = None):
        super().__init__(f"fal media upload failed: {code}")
        self.code = code
        self.stage = stage
        self.http_status = http_status
        self.outcome_unknown = outcome_unknown
        self.artifact = {**(artifact or {}), "stage": stage, "outcome_unknown": outcome_unknown}


class FalUploadUncertain(FalUploadError):
    """A media POST may have succeeded. Persist this state; do not retry blindly."""

    def __init__(self, code: str, *, stage: str = "upload", http_status: int | None = None,
                 artifact: dict | None = None):
        super().__init__(code, stage=stage, http_status=http_status,
                         outcome_unknown=True, artifact=artifact)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _header_value(value: object, *, limit: int = 8192) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= limit
            and all(32 < ord(char) < 127 for char in value))


def _key() -> str:
    value = get_setting("FAL_KEY").strip()
    if not _header_value(value):
        raise FalUploadError("missing_or_invalid_fal_key")
    return value


def _matches_mime(data: bytes, mime_type: str) -> bool:
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if mime_type == "video/mp4":
        return len(data) >= 12 and data[4:8] == b"ftyp"
    return False


def _reject_constant(_):
    raise ValueError("non_finite_json")


def _access_url(value: object) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(char) <= 32 or ord(char) >= 127 for char in value)):
        raise ValueError("invalid_access_url")
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.netloc != host
            or not re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)*fal\.media", host)
            or parsed.query or parsed.fragment or not parsed.path.startswith("/files/")
            or parsed.path == "/files/" or "/../" in parsed.path or "/./" in parsed.path):
        raise ValueError("invalid_access_url")
    return value


class FalMediaUploader:
    def __init__(self, *, transport=None, timeout: float = 30.0,
                 max_upload_bytes: int = MAX_UPLOAD_BYTES):
        if (type(timeout) not in (int, float) or not 0 < timeout <= 60
                or type(max_upload_bytes) is not int or not 0 < max_upload_bytes <= MAX_UPLOAD_BYTES):
            raise ValueError("Upload timeout must be 0..60 seconds; limit must be 1..100 MiB")
        self._transport = transport or build_opener(_NoRedirect()).open
        self.timeout = timeout
        self.max_upload_bytes = max_upload_bytes

    def preflight(self) -> None:
        """Validate the named process key only, without a network request."""
        _key()

    def _exchange(self, request: Request, *, stage: str, artifact: dict) -> dict:
        try:
            with self._transport(request, timeout=self.timeout) as response:
                if response.geturl() != request.full_url:
                    raise ValueError("redirected_response")
                if type(response.status) is not int or not 200 <= response.status < 300:
                    if type(response.status) is not int:
                        raise ValueError("invalid_status")
                    raise HTTPError(request.full_url, response.status, "HTTP error", response.headers, None)
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError("oversized_response")
                result = json.loads(raw, parse_constant=_reject_constant)
                if not isinstance(result, dict):
                    raise ValueError("invalid_response")
                return result
        except HTTPError as error:
            status = error.code
            error.close()
            unknown = stage == "upload" and (status >= 500 or status in (408, 301, 302, 303, 307, 308))
            cls = FalUploadUncertain if unknown else FalUploadError
            raise cls("http_error", stage=stage, http_status=status, artifact=artifact) from None
        except Exception:
            cls = FalUploadUncertain if stage == "upload" else FalUploadError
            raise cls("untrusted_or_failed_response", stage=stage, artifact=artifact) from None

    def upload(self, path: Path | str, mime_type: str, expected_sha256: str) -> dict:
        """Upload precisely the bytes whose hash was reviewed by the caller.

        The caller checks actual image/video decoding and reference clip duration.
        This boundary checks the signature, MIME, hash and cap before any network
        operation. Source bytes remain pinned in memory during the two requests.
        """
        if not isinstance(mime_type, str) or mime_type not in _SUFFIXES:
            raise FalUploadError("unsupported_mime_type")
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise FalUploadError("invalid_expected_sha256")
        try:
            source = Path(path).resolve(strict=True)
            if not source.is_file() or not 0 < source.stat().st_size <= self.max_upload_bytes:
                raise ValueError("invalid_source_size")
            with source.open("rb") as stream:
                data = stream.read(self.max_upload_bytes + 1)
        except Exception:
            raise FalUploadError("unreadable_or_oversized_source") from None
        if not 0 < len(data) <= self.max_upload_bytes:
            raise FalUploadError("unreadable_or_oversized_source")
        digest = hashlib.sha256(data).hexdigest()
        artifact = {"source_path": str(source), "source_sha256": digest,
                    "mime_type": mime_type, "size_bytes": len(data)}
        if digest != expected_sha256:
            raise FalUploadError("source_hash_mismatch", artifact=artifact)
        if not _matches_mime(data, mime_type):
            raise FalUploadError("source_mime_mismatch", artifact=artifact)
        key = _key()
        auth = self._exchange(Request(TOKEN_URL, data=b"{}", method="POST", headers={
            "Authorization": f"Key {key}", "Content-Type": "application/json", "Accept": "application/json",
        }), stage="token", artifact=artifact)
        try:
            token, token_type = auth["token"], auth["token_type"]
            base_url = _TOKEN_BASE_URLS.get(auth.get("base_url"))
            if (base_url is None
                    or not _header_value(token) or key in token
                    or not isinstance(token_type, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", token_type)):
                raise ValueError("invalid_token")
            expires_at = datetime.fromisoformat(auth["expires_at"])
            if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc):
                raise ValueError("expired_token")
        except Exception:
            raise FalUploadError("untrusted_token_response", stage="token", artifact=artifact) from None
        result = self._exchange(Request(base_url + "/files/upload", data=data, method="POST", headers={
            "Authorization": f"{token_type} {token}", "Content-Type": mime_type,
            "Accept": "application/json", "X-Fal-File-Name": f"{digest[:20]}{_SUFFIXES[mime_type]}",
        }), stage="upload", artifact=artifact)
        try:
            url = _access_url(result.get("access_url"))
            if key in url or token in url:
                raise ValueError("credential_in_access_url")
        except Exception:
            raise FalUploadUncertain("untrusted_access_url", artifact=artifact) from None
        return {**artifact, "access_url": url, "uploaded_at": datetime.now(timezone.utc).isoformat()}
