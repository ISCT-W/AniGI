"""Small, synchronous fal queue boundary; orchestration and authorization live above it.

Schema snapshot: 2026-09-07, official fal H3 Max OpenAPI.
No upload, polling loop, paid-call retry, or secret persistence happens here.
"""

from __future__ import annotations

import json
import os
from .settings import get_setting
import re
import tempfile
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ENDPOINTS = frozenset(
    f"minimax/h3-max/{kind}"
    for kind in ("text-to-video", "image-to-video", "reference-to-video")
)
QUEUE_ORIGIN = "https://queue.fal.run"
_COMMON = {
    "prompt", "prompt_expansion_mode", "duration", "resolution", "seed",
    "enable_safety_checker", "sync_mode",
}
_RATIOS = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
_REFERENCES = {"reference_image_urls": 9, "reference_video_urls": 3, "reference_audio_urls": 3}


class ProviderError(RuntimeError):
    """Sanitized provider failure, with an optional HTTP status for recovery."""

    def __init__(self, message: str, *, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


class SubmissionUncertain(ProviderError):
    """The POST may have been accepted. Reconcile manually; never resubmit blindly."""


def _https_url(value: object) -> object:
    # Reject parser-normalized control characters, credentials, ports, and fragments.
    if not isinstance(value, str) or not value or any(ord(c) <= 32 or ord(c) == 127 for c in value):
        raise ValueError("Expected a nonempty HTTPS URL without whitespace")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "https" and parsed.hostname and parsed.port in (None, 443)
            and not parsed.username and not parsed.password and not parsed.fragment
            and "\\" not in value
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Expected an HTTPS URL without credentials, custom ports, or fragments")
    return parsed


def request_duration(arguments: dict) -> int:
    duration = arguments.get("duration", 5)
    if type(duration) is not int or not 5 <= duration <= 15:
        raise ValueError("duration must be an integer from 5 to 15 seconds")
    return duration


def validate_request(endpoint: str, arguments: dict) -> None:
    """Validate the intentionally narrow local subset before spending or uploading.

    Clip duration/content behind reference URLs cannot be inferred from the URL;
    the orchestrator must inspect those assets before calling submit.
    """
    if not isinstance(endpoint, str) or endpoint not in ENDPOINTS:
        raise ValueError("Unsupported endpoint; only the three documented H3 Max queue endpoints are enabled")
    if not isinstance(arguments, dict) or any(not isinstance(k, str) for k in arguments):
        raise ValueError("arguments must be a JSON object with string keys")
    kind = endpoint.rsplit("/", 1)[1]
    allowed = _COMMON | ({"image_url", "end_image_url"} if kind == "image-to-video" else {"aspect_ratio"})
    if kind == "reference-to-video":
        allowed |= set(_REFERENCES)
    if set(arguments) - allowed:
        raise ValueError("Arguments include fields unsupported by this endpoint")
    prompt = arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or not 1 <= len(prompt) <= 50000:
        raise ValueError("prompt must contain 1 to 50000 characters")
    if arguments.get("prompt_expansion_mode") not in ("balanced", "quality"):
        raise ValueError("prompt_expansion_mode must be explicitly balanced or quality")
    request_duration(arguments)
    if arguments.get("resolution", "768P") not in ("480P", "768P"):
        raise ValueError("resolution must be 480P or 768P")
    if arguments.get("seed") is not None and type(arguments["seed"]) is not int:
        raise ValueError("seed must be an integer or null")
    for field in ("enable_safety_checker", "sync_mode"):
        if field in arguments and type(arguments[field]) is not bool:
            raise ValueError(f"{field} must be boolean")
    if arguments.get("sync_mode", False):
        raise ValueError("sync_mode=true is unsupported; persist a CDN result instead of base64")
    if "aspect_ratio" in arguments:
        choices = _RATIOS | ({"adaptive"} if kind == "reference-to-video" else set())
        if not isinstance(arguments["aspect_ratio"], str) or arguments["aspect_ratio"] not in choices:
            raise ValueError("Unsupported aspect_ratio")
    for field in ("image_url", "end_image_url"):
        if arguments.get(field) is not None:
            _https_url(arguments[field])
    total = 0
    for field, maximum in _REFERENCES.items():
        values = arguments.get(field, [])
        if not isinstance(values, list) or len(values) > maximum:
            raise ValueError(f"{field} must be an array with at most {maximum} URLs")
        for value in values:
            _https_url(value)
        total += len(values)
    if total > 12:
        raise ValueError("References must total at most 12 files")
    if arguments.get("reference_audio_urls") and not (
        arguments.get("reference_image_urls") or arguments.get("reference_video_urls")
    ):
        raise ValueError("Audio references require at least one image or video reference")


def _queue_url(value: object) -> str:
    parsed = _https_url(value)
    if parsed.hostname != "queue.fal.run":
        raise ValueError("Queue receipt URL must use queue.fal.run")
    # A receipt can use either the app root or full endpoint path. Use returned URLs.
    if not re.fullmatch(
        r"/minimax/h3-max(?:/(?:text-to-video|image-to-video|reference-to-video))?"
        r"/requests/[A-Za-z0-9_-]+(?:/(?:status|response|cancel))?", parsed.path
    ):
        raise ValueError("Queue receipt URL is outside the supported model request paths")
    if parsed.query not in ("", "logs=1", "logs=0"):
        raise ValueError("Unsupported queue URL query")
    return value


def _receipt_url(receipt: dict, field: str) -> str:
    if not isinstance(receipt, dict):
        raise ValueError("Expected a queue receipt")
    request_id = receipt.get("request_id")
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", request_id):
        raise ValueError("Invalid queue request_id")
    url = _queue_url(receipt.get(field))
    tail = urlsplit(url).path.split("/requests/", 1)[1].split("/")
    suffix = {"status_url": ["status"], "cancel_url": ["cancel"], "response_url": None}[field]
    if tail[0] != request_id or (suffix is not None and tail[1:] != suffix):
        raise ValueError("Receipt URL does not match its request_id or operation")
    if field == "response_url" and tail[1:] not in ([], ["response"]):
        raise ValueError("Invalid queue response URL")
    return url


def _media_url(url: str) -> str:
    parsed = _https_url(url)
    if parsed.hostname != "fal.media" and not parsed.hostname.endswith(".fal.media"):
        raise ValueError("Generated media download must use the fal.media CDN")
    return url


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class FalProvider:
    def __init__(self, *, transport=None, timeout: float = 30.0, max_download_bytes: int = 536870912):
        if not 0 < timeout <= 60 or type(max_download_bytes) is not int or max_download_bytes <= 0:
            raise ValueError("timeout must be 0..60 seconds and download limit must be positive")
        self._transport = transport or build_opener(_NoRedirect()).open
        self.timeout = timeout
        self.max_download_bytes = max_download_bytes

    def _json(self, method: str, url: str, arguments: dict | None = None) -> dict:
        key = get_setting("FAL_KEY").strip()
        if not key or any(ord(c) < 32 or ord(c) == 127 for c in key):
            raise ProviderError("FAL_KEY is missing or invalid in this process environment")
        data = json.dumps(arguments, ensure_ascii=False, allow_nan=False).encode() if arguments is not None else None
        request = Request(url, data=data, method=method, headers={
            "Authorization": f"Key {key}", "Content-Type": "application/json", "Accept": "application/json",
        })
        try:
            with self._transport(request, timeout=self.timeout) as response:
                if response.geturl() != url:
                    raise ProviderError("Unexpected queue redirect; receipt was not trusted")
                if not 200 <= response.status < 300:
                    raise HTTPError(url, response.status, "HTTP error", response.headers, None)
                raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ProviderError("Queue response exceeds local size limit")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ProviderError("Queue response is not a JSON object")
                return result
        except HTTPError as error:
            status = error.code
            error.close()
            cls = SubmissionUncertain if method == "POST" and (status >= 500 or status == 408) else ProviderError
            raise cls(f"fal returned HTTP {status}; response body withheld", http_status=status) from None
        except (URLError, OSError, TimeoutError, HTTPException):
            cls = SubmissionUncertain if method == "POST" else ProviderError
            raise cls("fal transport failed; submission outcome may require reconciliation" if method == "POST" else "fal transport failed") from None
        except (ValueError, ProviderError):
            cls = SubmissionUncertain if method == "POST" else ProviderError
            raise cls("fal returned an invalid or untrusted response; do not repeat a submission blindly") from None

    def submit(self, endpoint: str, arguments: dict) -> dict:
        validate_request(endpoint, arguments)
        receipt = self._json("POST", f"{QUEUE_ORIGIN}/{endpoint}", arguments)
        try:
            for field in ("status_url", "response_url", "cancel_url"):
                _receipt_url(receipt, field)
        except ValueError:
            raise SubmissionUncertain("fal submission receipt is incomplete or untrusted; reconcile in the dashboard before resubmitting") from None
        return receipt

    def status(self, receipt: dict) -> dict:
        result = self._json("GET", _receipt_url(receipt, "status_url"))
        if result.get("status") not in ("IN_QUEUE", "IN_PROGRESS", "COMPLETED"):
            raise ProviderError("Unknown queue status; retain the request for reconciliation")
        if result.get("request_id", receipt["request_id"]) != receipt["request_id"]:
            raise ProviderError("Queue status request_id mismatch")
        return result

    def result(self, receipt: dict) -> dict:
        # COMPLETED can carry a model failure; the caller must inspect error fields.
        return self._json("GET", _receipt_url(receipt, "response_url"))

    def cancel(self, receipt: dict) -> dict:
        # Acceptance only requests cancellation: it does not prove work or billing stopped.
        return self._json("PUT", _receipt_url(receipt, "cancel_url"))

    def download(self, url: str, path: Path) -> Path:
        """Stream a generated fal CDN file, atomically; refuse to overwrite a file."""
        url = _media_url(url)
        path = Path(path)
        if path.exists():
            raise FileExistsError("Download destination already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            for redirect in range(4):
                try:
                    response = self._transport(Request(url, headers={"Accept": "video/*"}), timeout=self.timeout)
                    break
                except HTTPError as error:
                    code, location = error.code, error.headers.get("Location")
                    error.close()
                    if code not in (301, 302, 303, 307, 308) or not location or redirect == 3:
                        raise ProviderError(f"Media download returned HTTP {code}", http_status=code) from None
                    url = _media_url(urljoin(url, location))
            with response:
                if response.geturl() != url or response.status != 200:
                    raise ProviderError("Unexpected media download response")
                size = response.headers.get("Content-Length")
                if size is not None and (not size.isdigit() or int(size) > self.max_download_bytes):
                    raise ProviderError("Media download exceeds local size limit or has invalid length")
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as output:
                    temporary = Path(output.name)
                    count = 0
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > self.max_download_bytes:
                            raise ProviderError("Media download exceeds local size limit")
                        output.write(chunk)
                    if count == 0 or (size is not None and count != int(size)):
                        raise ProviderError("Media download is empty or truncated")
                # Hard-link gives atomic no-clobber semantics even if another writer races.
                os.link(temporary, path)
                return path
        except (URLError, OSError, TimeoutError, HTTPException) as error:
            if isinstance(error, FileExistsError):
                raise
            raise ProviderError("Media download failed; no complete artifact was published") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
