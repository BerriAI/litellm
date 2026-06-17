"""Mistral OCR plugin.

Thin layer that conforms to `LLMPlugin` by way of `TransformingLLMPlugin`.
The transform_request step normalizes the incoming JSON into Mistral's OCR
request shape; call_upstream hits Mistral's OCR endpoint; transform_response
maps Mistral's response (or HTTP error) into our interface shape.

The plugin does not import any core internals; it only depends on `base.py`,
which the core also does not import. Keeping the plugin in its own process
makes this isolation automatic.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Final

from .base import (
    Capabilities,
    Err,
    Ok,
    PluginError,
    PluginRequest,
    PluginResponse,
    TransformingLLMPlugin,
)

MISTRAL_OCR_URL: Final[str] = "https://api.mistral.ai/v1/ocr"
SUPPORTED_MODELS: Final[tuple[str, ...]] = (
    "mistral-ocr-latest",
    "mistral-ocr-2505",
)


@dataclass(frozen=True, slots=True)
class MistralOCRRequest:
    payload: bytes


@dataclass(frozen=True, slots=True)
class MistralOCRResponse:
    body: bytes
    content_type: str


class MistralOCRPlugin(TransformingLLMPlugin[MistralOCRRequest, MistralOCRResponse]):
    def __init__(
        self,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        upstream_url: str = MISTRAL_OCR_URL,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. The Mistral OCR plugin requires an API "
                "key in the environment; never hardcode it."
            )
        self._api_key = key
        self._timeout_s = timeout_s
        self._upstream_url = upstream_url

    def capabilities(self) -> Capabilities:
        return Capabilities(models=SUPPORTED_MODELS, endpoints=("/handle",))

    def transform_request(self, request: PluginRequest) -> MistralOCRRequest:
        return MistralOCRRequest(payload=request.body)

    def call_upstream(
        self, native_request: MistralOCRRequest
    ) -> Ok[MistralOCRResponse] | Err:
        req = urllib.request.Request(
            self._upstream_url,
            data=native_request.payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                body = resp.read()
                ct = resp.headers.get("Content-Type", "application/json")
                return Ok(MistralOCRResponse(body=body, content_type=ct))
        except urllib.error.HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            return Err(
                PluginError(
                    code=f"mistral_http_{e.code}",
                    message=_safe_error_message(raw, default=str(e)),
                    type="upstream_http_error",
                    http_status=502 if e.code >= 500 else e.code,
                )
            )
        except urllib.error.URLError as e:
            return Err(
                PluginError(
                    code="mistral_network_error",
                    message=str(e.reason) if hasattr(e, "reason") else str(e),
                    type="upstream_transport_error",
                    http_status=502,
                )
            )
        except TimeoutError as e:
            return Err(
                PluginError(
                    code="mistral_timeout",
                    message=str(e) or "timeout",
                    type="upstream_timeout",
                    http_status=504,
                )
            )

    def transform_response(self, native_response: MistralOCRResponse) -> PluginResponse:
        return PluginResponse(
            body=native_response.body,
            content_type=native_response.content_type,
            status_code=200,
        )


def _safe_error_message(raw: bytes, default: str) -> str:
    if not raw:
        return default
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")[:500]
    if isinstance(parsed, dict):
        for key in ("message", "error", "detail"):
            val = parsed.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                inner = val.get("message")
                if isinstance(inner, str):
                    return inner
    return raw.decode("utf-8", errors="replace")[:500]
