"""Python preparation, encoding, and response transforms for retained OCR calls."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Final, cast  # noqa: TID251  # native callables and legacy header types require boundary casts

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig, DocumentType, OCRResponse
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.rust_bridge.timeouts import timeout_to_seconds

OCRRoots = tuple[dict[str, object], str, dict[str, object], None]
OCRWire = tuple[int, list[tuple[bytes, bytes]], bytes]
OCREncoded = tuple[str, list[tuple[bytes, bytes]], bytes, float]


def retained_timeout_seconds(timeout: float | httpx.Timeout) -> float | None:
    seconds: Final = timeout_to_seconds(timeout)
    return seconds if seconds is not None and math.isfinite(seconds) and seconds > 0 else None


@dataclass(kw_only=True, slots=True)
class OCRRetainedBoundary:
    handler: BaseLLMHTTPHandler
    model: str
    document: DocumentType
    optional_params: dict[str, object]
    logging_obj: Logging
    api_key: str | None
    api_base: str | None
    headers: dict[str, object] | None
    provider_config: BaseOCRConfig
    litellm_params: dict[str, object]
    custom_llm_provider: str
    timeout: float | httpx.Timeout
    client: HTTPHandler | AsyncHTTPHandler | None = None
    request: httpx.Request | None = field(default=None, init=False)

    def prepare(self) -> OCRRoots:
        roots: Final = self.handler._prepare_ocr_request(
            model=self.model,
            document=self.document,
            optional_params=self.optional_params,
            logging_obj=self.logging_obj,
            api_key=self.api_key,
            api_base=self.api_base,
            headers=self.headers,
            provider_config=self.provider_config,
            litellm_params=self.litellm_params,
        )
        if not isinstance(self.client, HTTPHandler):
            self.client = _get_httpx_client()
        return roots

    async def aprepare(self) -> OCRRoots:
        roots: Final = await self.handler._async_prepare_ocr_request(
            model=self.model,
            document=self.document,
            optional_params=self.optional_params,
            logging_obj=self.logging_obj,
            api_key=self.api_key,
            api_base=self.api_base,
            headers=self.headers,
            provider_config=self.provider_config,
            litellm_params=self.litellm_params,
        )
        if not isinstance(self.client, AsyncHTTPHandler):
            self.client = get_async_httpx_client(llm_provider=litellm.LlmProviders(self.custom_llm_provider))
        return roots

    def encode(self, roots: OCRRoots) -> OCREncoded:
        headers, url, data, _files = roots
        seconds: Final = retained_timeout_seconds(self.timeout)
        if seconds is None:
            raise ValueError("Retained OCR requires a positive finite read timeout")
        if self.client is None:
            raise RuntimeError("Retained OCR must be prepared before encoding")
        try:
            self.request = self.client.client.build_request(
                "POST",
                url,
                headers=cast(dict[str, str], headers),
                json=data,
                timeout=self.timeout,
            )
            return str(self.request.url), self.request.headers.raw, self.request.read(), seconds
        except Exception as e:  # noqa: BLE001  # match the Python OCR handler's encoding error mapping
            raise self.handler._handle_error(e=e, provider_config=self.provider_config)

    def _response(self, wire: OCRWire) -> httpx.Response:
        if self.request is None:
            raise RuntimeError("Retained OCR must be encoded before finishing")
        status, headers, content = wire
        try:
            response: Final = httpx.Response(status, headers=headers, content=content, request=self.request)
            response.raise_for_status()
        except Exception as e:  # noqa: BLE001  # match the Python OCR handler's response error mapping
            raise self.handler._handle_error(e=e, provider_config=self.provider_config)
        return response

    def finish(self, wire: OCRWire) -> OCRResponse:
        return self.handler._transform_ocr_response(
            provider_config=self.provider_config,
            model=self.model,
            response=self._response(wire),
            logging_obj=self.logging_obj,
            optional_params=self.optional_params,
        )

    async def afinish(self, wire: OCRWire) -> OCRResponse:
        return await self.provider_config.async_transform_ocr_response(
            model=self.model,
            raw_response=self._response(wire),
            logging_obj=self.logging_obj,
            optional_params=self.optional_params,
        )


RustOCRRetained = Callable[[OCRRetainedBoundary], OCRResponse]
RustAOCRRetained = Callable[[OCRRetainedBoundary], Awaitable[OCRResponse]]


def load_rust_ocr_retained() -> RustOCRRetained | None:
    from litellm.rust_bridge import get_native_bridge

    native: Final = get_native_bridge()
    return cast(RustOCRRetained | None, getattr(native, "ocr_retained", None))


def load_rust_aocr_retained() -> RustAOCRRetained | None:
    from litellm.rust_bridge import get_native_bridge

    native: Final = get_native_bridge()
    return cast(RustAOCRRetained | None, getattr(native, "aocr_retained", None))
