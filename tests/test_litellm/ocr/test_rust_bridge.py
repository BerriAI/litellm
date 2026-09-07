"""Tests for the optional Rust-backed OCR path."""

import asyncio
import builtins
import contextvars
import copy
import gc
import importlib
import inspect
import os
import subprocess
import sys
import threading
import types
import weakref
from dataclasses import replace
from datetime import datetime
from typing import Any, Final, cast

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.ocr.transformation import OCRRequestData, OCRResponse
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.rust_bridge import configuration

# `litellm/__init__.py` does `from .ocr.main import *`, which binds the `ocr`
# function onto `litellm.ocr` and shadows the submodule, so import the modules
# explicitly via importlib rather than attribute traversal.
ocr_main = importlib.import_module("litellm.ocr.main")
rust_bridge = importlib.import_module("litellm.rust_bridge.ocr")
rust_bridge_bindings = importlib.import_module("litellm.rust_bridge.bindings")
rust_bridge_loader = importlib.import_module("litellm.rust_bridge.loader")

MODEL = "mistral/mistral-ocr-latest"
DOCUMENT: dict[str, object] = {
    "type": "document_url",
    "document_url": "https://example.com/doc.pdf",
}

FAKE_OCR_RESPONSE: dict[str, object] = {
    "pages": [{"index": 0, "markdown": "hello world"}],
    "model": "mistral-ocr-2505-completion",
    "document_annotation": None,
    "usage_info": {"pages_processed": 1},
    "object": "ocr",
}


class CapturedException(Exception):
    pass


class RecordingBridge:
    """A fake ``RustOcr`` callable that records the boundary it was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, boundary: rust_bridge.OCRBoundary) -> OCRResponse:
        self.calls.append(
            {
                "model": boundary.model,
                "document": boundary.document,
                "api_key": boundary.api_key,
                "api_base": boundary.api_base,
                "custom_llm_provider": boundary.custom_llm_provider,
                "extra_headers": boundary.headers,
                "optional_params": boundary.optional_params,
                "timeout": boundary.timeout,
            }
        )
        return OCRResponse.model_validate(FAKE_OCR_RESPONSE)


class RecordingAsyncBridge:
    """A fake async ``RustAocr`` callable that records the boundary it was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(self, boundary: rust_bridge.OCRBoundary) -> OCRResponse:
        self.calls.append(
            {
                "model": boundary.model,
                "document": boundary.document,
                "api_key": boundary.api_key,
                "api_base": boundary.api_base,
                "custom_llm_provider": boundary.custom_llm_provider,
                "extra_headers": boundary.headers,
                "optional_params": boundary.optional_params,
                "timeout": boundary.timeout,
            }
        )
        return OCRResponse.model_validate(FAKE_OCR_RESPONSE)


class RaisingBridge:
    def __call__(self, boundary: rust_bridge.OCRBoundary) -> OCRResponse:
        raise RuntimeError("bridge failed")


class RaisingAsyncBridge:
    async def __call__(self, boundary: rust_bridge.OCRBoundary) -> OCRResponse:
        raise RuntimeError("bridge failed")


class RecordingLogging:
    """A spy standing in for ``LiteLLMLoggingObj`` to capture ``pre_call``."""

    def __init__(self) -> None:
        self.pre_call_kwargs: dict[str, object] | None = None

    def pre_call(
        self,
        *,
        input: str,
        api_key: str | None,
        additional_args: dict[str, object],
    ) -> None:
        self.pre_call_kwargs = {
            "input": input,
            "api_key": api_key,
            "additional_args": additional_args,
        }


class FakeOCRConfig:
    """A stand-in ``BaseOCRConfig`` that echoes the request it would build."""

    def __init__(self, api_key_env_var: str = "MISTRAL_API_KEY") -> None:
        self.api_key_env_var = api_key_env_var
        self.seen_api_keys: list[str | None] = []

    def get_api_key_env_var(self) -> str:
        return self.api_key_env_var

    def validate_environment(
        self,
        *,
        headers: dict[str, object],
        model: str,
        api_key: str | None,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> dict[str, object]:
        self.seen_api_keys.append(api_key)
        return {"Authorization": f"Bearer {api_key}", **headers}

    def get_complete_url(
        self,
        *,
        api_base: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
    ) -> str:
        return f"{api_base or 'https://api.mistral.ai/v1'}/ocr"

    def transform_ocr_request(
        self,
        *,
        model: str,
        document: dict[str, object],
        optional_params: dict[str, object],
        headers: dict[str, object],
        api_key: str | None,
        api_base: str | None,
    ) -> OCRRequestData:
        return OCRRequestData(data={"model": model, "document": document, **optional_params}, files=None)

    async def async_transform_ocr_request(
        self,
        *,
        model: str,
        document: dict[str, object],
        optional_params: dict[str, object],
        headers: dict[str, object],
        api_key: str | None,
        api_base: str | None,
    ) -> OCRRequestData:
        return self.transform_ocr_request(
            model=model,
            document=document,
            optional_params=optional_params,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
        )


def build_prepared_request(
    *,
    logging_obj: RecordingLogging | None = None,
    provider_config: FakeOCRConfig | None = None,
    model: str = "mistral-ocr-latest",
    document: dict[str, object] = DOCUMENT,
    api_key: str | None = "sk-test",
    api_base: str | None = None,
    custom_llm_provider: str = "mistral",
    extra_headers: dict[str, object] | None = None,
    optional_params: dict[str, object] | None = None,
    litellm_params: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = 12.5,
) -> Any:
    from litellm.constants import request_timeout

    return ocr_main._PreparedOCRRequest(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        provider_config=provider_config or FakeOCRConfig(),
        optional_params=optional_params or {},
        litellm_params=litellm_params or {},
        effective_timeout=timeout if timeout is not None else float(request_timeout),
        litellm_logging_obj=logging_obj or RecordingLogging(),
    )


class BoundaryDriver:
    """Stands in for the native route: drives the boundary's own methods."""

    def __init__(self) -> None:
        self.roots: rust_bridge.OCRRoots | None = None
        self.encoded: rust_bridge.OCREncoded | None = None

    def __call__(self, boundary: rust_bridge.OCRBoundary) -> OCRResponse:
        self.roots = boundary.prepare()
        self.encoded = boundary.encode(self.roots)
        return OCRResponse.model_validate(FAKE_OCR_RESPONSE)


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    """Keep the global toggle isolated between tests."""
    rust_bridge._OCR.reset()
    rust_bridge._AOCR.reset()
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL
    yield
    rust_bridge._OCR.reset()
    rust_bridge._AOCR.reset()
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL


@pytest.fixture
def fake_bridge():
    """Enable the Rust path with an injected recording bridge (no native wheel)."""
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    return bridge


@pytest.fixture
def fake_async_bridge():
    """Enable the async Rust path with an injected recording bridge."""
    bridge = RecordingAsyncBridge()
    litellm.rust(True)
    rust_bridge._AOCR.override(bridge)
    return bridge


def test_load_rust_ocr_returns_injected_impl():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    assert rust_bridge.load_rust_ocr() is bridge


def test_native_bridge_loader_returns_none_when_extension_absent(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "litellm.rust_bridge" and "_native" in fromlist:
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert rust_bridge_loader.get_native_bridge() is None


def test_native_bridge_loader_caches_absent_extension(monkeypatch):
    real_import = builtins.__import__
    attempts = 0

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal attempts
        if name == "litellm.rust_bridge" and "_native" in fromlist:
            attempts += 1
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert rust_bridge_loader.get_native_bridge() is None
    assert rust_bridge_loader.get_native_bridge() is None
    assert attempts == 1


def test_native_bridge_loader_reset_forces_relookup(monkeypatch):
    real_import = builtins.__import__
    attempts = 0

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal attempts
        if name == "litellm.rust_bridge" and "_native" in fromlist:
            attempts += 1
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert rust_bridge_loader.get_native_bridge() is None
    rust_bridge_loader.reset_native_bridge_cache()
    assert rust_bridge_loader.get_native_bridge() is None
    assert attempts == 2


def test_native_bridge_available_reflects_loader(monkeypatch):
    fake_module = types.ModuleType("litellm.rust_bridge._native")
    monkeypatch.setattr(rust_bridge_loader, "get_native_bridge", lambda: fake_module)

    assert rust_bridge_loader.native_bridge_available() is True


def test_load_rust_aocr_returns_injected_impl():
    bridge = RecordingAsyncBridge()
    litellm.rust(True)
    rust_bridge._AOCR.override(bridge)
    assert rust_bridge.load_rust_aocr() is bridge


def test_toggle_without_ocr_arg_preserves_injected_impl():
    """The public flag must not clobber an internal test binding."""
    bridge = RecordingBridge()
    async_bridge = RecordingAsyncBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    rust_bridge._AOCR.override(async_bridge)

    litellm.rust(False)
    assert rust_bridge.load_rust_ocr() is bridge
    assert rust_bridge.load_rust_aocr() is async_bridge
    litellm.rust(True)
    assert rust_bridge.load_rust_ocr() is bridge
    assert rust_bridge.load_rust_aocr() is async_bridge


def test_explicit_ocr_none_clears_injected_impl(monkeypatch):
    monkeypatch.setattr(
        rust_bridge_bindings,
        "get_native_bridge",
        lambda: None,
    )
    bridge = RecordingBridge()
    async_bridge = RecordingAsyncBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    rust_bridge._AOCR.override(async_bridge)

    rust_bridge._OCR.override(None)
    rust_bridge._AOCR.override(None)
    assert rust_bridge.load_rust_ocr() is None
    assert rust_bridge.load_rust_aocr() is None


def test_load_rust_ocr_none_when_extension_absent(monkeypatch):
    """With no injected impl and no compiled wheel, the loader returns None so the
    caller degrades to the Python path instead of raising ImportError."""
    monkeypatch.setattr(
        rust_bridge_bindings,
        "get_native_bridge",
        lambda: None,
    )
    litellm.rust(True)  # no impl injected; extension isn't built in CI
    assert rust_bridge.load_rust_ocr() is None
    assert rust_bridge.load_rust_aocr() is None


def test_load_rust_ocr_uses_compiled_extension(monkeypatch):
    """With no injected impl but a packaged ``litellm.rust_bridge._native`` importable,
    the loader returns the extension's ``ocr`` callable. The native wheel isn't
    built in CI, so stand in a fake module via the bridge loader."""
    fake_module = types.ModuleType("litellm.rust_bridge._native")
    fake_module.ocr = lambda **kwargs: dict(FAKE_OCR_RESPONSE)  # type: ignore[attr-defined]
    fake_module.aocr = lambda **kwargs: dict(FAKE_OCR_RESPONSE)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        rust_bridge_bindings,
        "get_native_bridge",
        lambda: fake_module,
    )

    litellm.rust(True)  # enabled, no impl injected -> import the extension
    assert rust_bridge.load_rust_ocr() is fake_module.ocr
    assert rust_bridge.load_rust_aocr() is fake_module.aocr


def test_timeout_to_seconds_handles_float_timeout_and_none():
    assert rust_bridge._timeout_to_seconds(12.5) == 12.5
    assert rust_bridge._timeout_to_seconds(None) is None
    assert rust_bridge._timeout_to_seconds(httpx.Timeout(30.0, read=42.0)) == 42.0


def test_run_rust_ocr_forwards_boundary_fields():
    bridge = RecordingBridge()
    logging_obj = RecordingLogging()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    response = ocr_main._run_rust_ocr(
        build_prepared_request(
            logging_obj=logging_obj,
            api_base="https://proxy.internal",
            extra_headers={"x-trace-id": "trace-1"},
            optional_params={"include_image_base64": True},
            timeout=12.5,
        )
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    assert bridge.calls[0] == {
        "model": "mistral-ocr-latest",
        "document": DOCUMENT,
        "api_key": "sk-test",
        "api_base": "https://proxy.internal",
        "custom_llm_provider": "mistral",
        "extra_headers": {"x-trace-id": "trace-1"},
        "optional_params": {"include_image_base64": True},
        "timeout": 12.5,
    }
    assert logging_obj.pre_call_kwargs is None  # pre_call now runs inside the boundary


def test_run_rust_ocr_passes_raw_api_key_to_provider_config():
    """Key resolution moved into provider ``validate_environment``; the boundary
    forwards the caller's key unchanged."""
    provider_config = FakeOCRConfig(api_key_env_var="PROVIDER_OCR_API_KEY")
    driver = BoundaryDriver()
    litellm.rust(True)
    rust_bridge._OCR.override(driver)

    ocr_main._run_rust_ocr(build_prepared_request(provider_config=provider_config, api_key="sk-explicit", timeout=None))
    ocr_main._run_rust_ocr(build_prepared_request(provider_config=provider_config, api_key=None, timeout=None))

    assert provider_config.seen_api_keys == ["sk-explicit", None]


def test_run_rust_ocr_preserves_native_response_identity():
    """The bridge returns the boundary's own finish() object, not a re-validated copy."""
    sentinel = OCRResponse.model_validate(FAKE_OCR_RESPONSE)

    class IdentityBridge:
        def __call__(self, boundary: rust_bridge.OCRBoundary) -> OCRResponse:
            return sentinel

    litellm.rust(True)
    rust_bridge._OCR.override(IdentityBridge())

    response = ocr_main._run_rust_ocr(build_prepared_request())

    assert response is sentinel


def test_boundary_prepare_runs_pre_call_and_encodes_the_same_roots():
    """The logging view must alias the execution roots: mutating the headers the
    callback received must surface in the encoded wire headers, while replacing
    a view field must not."""
    logging_obj = RecordingLogging()
    driver = BoundaryDriver()
    litellm.rust(True)
    rust_bridge._OCR.override(driver)

    seen: dict[str, object] = {}

    original_pre_call = logging_obj.pre_call

    def observing_pre_call(**kwargs: object) -> None:
        original_pre_call(**kwargs)
        view = cast(dict[str, object], kwargs["additional_args"])
        seen["headers"] = view["headers"]
        cast(dict[str, object], view["headers"])["X-Proof"] = "mutated"

    logging_obj.pre_call = observing_pre_call  # type: ignore[method-assign]

    ocr_main._run_rust_ocr(build_prepared_request(logging_obj=logging_obj, api_base="https://api.mistral.ai/v1"))

    assert logging_obj.pre_call_kwargs is not None
    additional_args = logging_obj.pre_call_kwargs["additional_args"]
    assert additional_args["api_base"] == "https://api.mistral.ai/v1/ocr"
    assert additional_args["headers"] == {"Authorization": "Bearer sk-test", "X-Proof": "mutated"}
    assert driver.roots is not None
    roots_headers, _url, _data, _files = driver.roots
    assert roots_headers is seen["headers"]
    assert driver.encoded is not None
    encoded_headers = {name.lower(): value for name, value in driver.encoded[1]}
    assert encoded_headers[b"x-proof"] == b"mutated"


def test_ocr_routes_to_rust_when_enabled(fake_bridge):
    response = litellm.ocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
        extra_headers={"x-trace-id": "trace-1"},
        include_image_base64=True,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    assert len(fake_bridge.calls) == 1
    call = fake_bridge.calls[0]
    assert call["model"] == "mistral-ocr-latest"
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    assert call["custom_llm_provider"] == "mistral"
    assert call["extra_headers"] == {"x-trace-id": "trace-1"}
    assert call["optional_params"].get("include_image_base64") is True


def test_ocr_routes_azure_ai_to_rust_when_enabled(fake_bridge):
    response = litellm.ocr(
        model="azure_ai/pixtral-12b-2409",
        document=DOCUMENT,
        api_key="sk-test",
        api_base="https://example.services.ai.azure.com",
    )

    assert isinstance(response, OCRResponse)
    assert len(fake_bridge.calls) == 1
    assert fake_bridge.calls[0]["model"] == "pixtral-12b-2409"
    assert fake_bridge.calls[0]["custom_llm_provider"] == "azure_ai"


def test_ocr_rust_path_converts_file_document_before_bridge(fake_bridge):
    response = litellm.ocr(
        model=MODEL,
        document={"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"},
        api_key="sk-test",
    )

    assert isinstance(response, OCRResponse)
    document = fake_bridge.calls[0]["document"]
    assert document["type"] == "document_url"
    assert document["document_url"].startswith("data:application/pdf;base64,")


def test_ocr_exception_type_uses_resolved_provider_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_exception_type(**kwargs: object) -> CapturedException:
        captured.update(kwargs)
        return CapturedException("wrapped")

    monkeypatch.setattr(ocr_main.litellm, "exception_type", fake_exception_type)
    litellm.rust(True)
    rust_bridge._OCR.override(RaisingBridge())

    with pytest.raises(CapturedException):
        litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured["model"] == "mistral-ocr-latest"
    assert captured["custom_llm_provider"] == "mistral"


@pytest.mark.asyncio
async def test_aocr_routes_to_async_rust_when_enabled(fake_async_bridge):
    response = await litellm.aocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
        extra_headers={"x-trace-id": "trace-1"},
        include_image_base64=True,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    assert len(fake_async_bridge.calls) == 1
    call = fake_async_bridge.calls[0]
    assert call["model"] == "mistral-ocr-latest"
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    assert call["custom_llm_provider"] == "mistral"
    assert call["extra_headers"] == {"x-trace-id": "trace-1"}
    assert call["optional_params"].get("include_image_base64") is True


@pytest.mark.asyncio
async def test_aocr_exception_type_uses_resolved_provider_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_exception_type(**kwargs: object) -> CapturedException:
        captured.update(kwargs)
        return CapturedException("wrapped")

    monkeypatch.setattr(ocr_main.litellm, "exception_type", fake_exception_type)
    litellm.rust(True)
    rust_bridge._AOCR.override(RaisingAsyncBridge())

    with pytest.raises(CapturedException):
        await litellm.aocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured["model"] == "mistral-ocr-latest"
    assert captured["custom_llm_provider"] == "mistral"


def test_ocr_forwards_timeout_to_rust(fake_bridge):
    """Caller-supplied timeout must flow into the Rust bridge so the fixed 600s
    client ceiling doesn't silently override shorter deadlines."""
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test", timeout=12.5)

    assert fake_bridge.calls[0]["timeout"] == 12.5


def test_ocr_passes_default_request_timeout_to_rust(fake_bridge):
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    from litellm.constants import request_timeout

    assert fake_bridge.calls[0]["timeout"] == float(request_timeout)


def test_ocr_does_not_route_to_rust_when_disabled():
    """With the flag off, the bridge must not be consulted even if an impl exists."""
    bridge = RecordingBridge()
    litellm.rust(False)
    rust_bridge.set_rust_ocr(ocr=bridge)

    assert rust_bridge.rust_ocr_enabled() is False
    # The impl stays available for injection, but the disabled flag gates usage,
    # so ocr() never reaches the Rust path (asserted via the enabled-path test).
    assert bridge.calls == []


def test_ocr_falls_back_to_python_when_bridge_unavailable(monkeypatch):
    """Rust enabled but no bridge available (no injected impl, no compiled wheel):
    ocr() must degrade to the Python HTTP handler instead of raising."""
    monkeypatch.setattr(rust_bridge, "load_rust_ocr", lambda: None)
    litellm.rust(True)  # enabled, but load_rust_ocr() returns None in CI

    captured = {}

    def fake_handler_ocr(**kwargs):
        captured["called"] = True
        return OCRResponse(pages=[], model="mistral-ocr-latest", object="ocr")

    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", fake_handler_ocr)

    response = litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured.get("called") is True  # Python path was used
    assert isinstance(response, OCRResponse)


def test_ocr_provider_configs_expose_api_key_env_vars():
    from litellm.llms.azure_ai.ocr.document_intelligence.transformation import (
        AzureDocumentIntelligenceOCRConfig,
    )
    from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig
    from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig
    from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
    from litellm.llms.vertex_ai.ocr.deepseek_transformation import (
        VertexAIDeepSeekOCRConfig,
    )
    from litellm.llms.vertex_ai.ocr.transformation import VertexAIOCRConfig

    assert BaseOCRConfig().get_api_key_env_var() is None
    assert MistralOCRConfig().get_api_key_env_var() == "MISTRAL_API_KEY"
    assert AzureAIOCRConfig().get_api_key_env_var() == "AZURE_AI_API_KEY"
    assert AzureDocumentIntelligenceOCRConfig().get_api_key_env_var() == "AZURE_DOCUMENT_INTELLIGENCE_API_KEY"
    assert VertexAIOCRConfig().get_api_key_env_var() == "VERTEX_AI_API_KEY"
    assert VertexAIDeepSeekOCRConfig().get_api_key_env_var() == "VERTEX_AI_API_KEY"


#################################################
# Proof: pre_call callbacks run against the same objects the wire request is
# built from, through the real native route, compared with the Python route.
#################################################

MISTRAL_OCR_RESPONSE_JSON: Final = (
    b'{"pages": [{"index": 0, "markdown": "proof"}], "model": "mistral-ocr-2505-completion",'
    b' "document_annotation": null, "usage_info": {"pages_processed": 1}, "object": "ocr"}'
)


class _WireRecorder:
    """Loopback OCR server recording every request it serves."""

    def __init__(self) -> None:
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.requests: list[dict[str, object]] = []
        self.received = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.release.set()
        self.status = 200
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # http.server API
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                recorder.requests.append(
                    {
                        "path": self.path,
                        "headers": {name.lower(): value for name, value in self.headers.items()},
                        "body": json.loads(body),
                    }
                )
                recorder.received.set()
                try:
                    if not recorder.release.wait(timeout=10):
                        return
                    self.send_response(recorder.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(MISTRAL_OCR_RESPONSE_JSON)))
                    self.end_headers()
                    self.wfile.write(MISTRAL_OCR_RESPONSE_JSON)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    recorder.finished.set()

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def api_base(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def wire_recorder():
    recorder = _WireRecorder()
    try:
        yield recorder
    finally:
        recorder.stop()


def _native_boundary_route_available() -> bool:
    bridge = rust_bridge_loader.get_native_bridge()
    if bridge is None:
        return False
    ocr_fn = getattr(bridge, "ocr", None)
    if ocr_fn is None:
        return False
    try:
        return str(inspect.signature(ocr_fn)) == "(boundary)"
    except (TypeError, ValueError):
        return False


@pytest.fixture
def native_ocr():
    if not _native_boundary_route_available():
        if os.environ.get("LITELLM_REQUIRE_NATIVE_OCR") == "1":
            pytest.fail("native OCR boundary route not built")
        pytest.skip("native OCR boundary route not built")
    return rust_bridge_loader.get_native_bridge()


class ProofCallback(CustomLogger):
    def __init__(self, document, context, events, fail=False) -> None:
        self.document = document
        self.context = context
        self.events = events
        self.fail = fail
        self.headers: dict[str, object] | None = None
        self.body: dict[str, object] | None = None
        self.details = None
        self.calls = 0

    def log_pre_api_call(self, model, messages, kwargs):
        self.calls += 1
        self.events.append(("mutate", self.context.get(), threading.get_ident(), asyncio.current_task()))
        self.context.set("callback")
        self.details = kwargs
        view = kwargs["additional_args"]
        self.headers = view["headers"]
        self.body = view["complete_input_dict"]
        self.headers["X-Proof"] = "mutated"
        self.body["document"]["document_url"] = "https://example.invalid/mutated-by-callback.pdf"
        self.document["document_name"] = "closure-mutation"
        view["headers"] = {"X-Replacement": "must-not-reach-wire"}
        view["complete_input_dict"] = {"model": "logging-only"}
        if self.fail:
            raise RuntimeError("expected pre-call failure")
        return {"additional_args": {"headers": {"X-Return": "ignored"}}}


def _assert_retained_wire(request: dict[str, object]) -> None:
    assert request["path"] == "/v1/ocr"
    headers = request["headers"]
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["x-proof"] == "mutated"
    assert "x-replacement" not in headers
    assert "x-return" not in headers
    body = request["body"]
    assert body["model"] == "mistral-ocr-latest"
    assert body["document"]["document_url"] == "https://example.invalid/mutated-by-callback.pdf"
    assert body["document"]["document_name"] == "closure-mutation"


async def _pre_call_contract(native, wire_recorder, monkeypatch, asynchronous, enabled, fail=False, control="original"):
    document = {
        "type": "document_url",
        "document_url": "https://example.invalid/original.pdf",
    }
    context = contextvars.ContextVar("ocr-pre-call", default="caller")
    events = []
    observations = []
    native_calls = []
    proof = ProofCallback(document, context, events, fail=fail)

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            events.append(("observe", context.get(), threading.get_ident(), asyncio.current_task()))
            view = kwargs["additional_args"]
            observations.append(
                (
                    kwargs is proof.details,
                    tuple(view["headers"].items()),
                    view["complete_input_dict"]["model"],
                    document["document_url"],
                )
            )

    def selected(boundary):
        if control == "copy-input":
            return replace(boundary, document=copy.deepcopy(boundary.document))
        if control in {"logging-roots", "tuple-only"}:

            class EncodeControl:
                def __getattr__(self, name):
                    return getattr(boundary, name)

                def encode(self, roots):
                    headers, url, body, files = roots
                    if control == "logging-roots":
                        view = boundary.logging_obj.model_call_details["additional_args"]
                        return boundary.encode((view["headers"], url, view["complete_input_dict"], files))
                    return boundary.encode((headers, url, body, files))

            return EncodeControl()
        return boundary

    def sync_call(boundary):
        native_calls.append("sync")
        if control == "duplicate-prepare":
            boundary.prepare()
        return native.ocr(selected(boundary))

    async def async_call(boundary):
        native_calls.append("async")
        if control == "duplicate-prepare":
            await boundary.aprepare()
        if control == "new-task":
            return await asyncio.create_task(native.aocr(selected(boundary)))
        return await native.aocr(selected(boundary))

    rust_bridge._OCR.override(sync_call)
    rust_bridge._AOCR.override(async_call)
    monkeypatch.setattr(litellm, "input_callback", [proof, Observe()])
    litellm.rust(enabled)
    caller = (threading.get_ident(), asyncio.current_task())
    arguments = dict(model=MODEL, document=document, api_key="sk-test", api_base=wire_recorder.api_base, num_retries=0)
    response = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert native_calls == (["async" if asynchronous else "sync"] if enabled else []), "native dispatch"
    assert isinstance(response, OCRResponse) and response.pages[0].markdown == "proof"
    assert proof.calls == 1, "callback count"
    assert proof.body is not None and proof.body["document"] is document, "caller identity"
    assert events == [("mutate", "caller", *caller), ("observe", "callback", *caller)], "callback context/order"
    assert context.get() == "callback", "caller context write"
    assert observations == [
        (
            True,
            (("X-Replacement", "must-not-reach-wire"),),
            "logging-only",
            "https://example.invalid/mutated-by-callback.pdf",
        )
    ], "observation-time values"
    assert len(wire_recorder.requests) == 1, "POST count"
    assert wire_recorder.requests[0]["headers"].get("x-proof") == "mutated", "execution roots"
    _assert_retained_wire(wire_recorder.requests[0])
    proof.body["document"]["document_url"] = "https://example.invalid/after-encode.pdf"
    assert document["document_url"] == "https://example.invalid/after-encode.pdf"
    assert (
        wire_recorder.requests[0]["body"]["document"]["document_url"]
        == "https://example.invalid/mutated-by-callback.pdf"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("enabled", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("fail", [False, True], ids=["return-ignored", "caught-error"])
async def test_pre_call_contract(native_ocr, wire_recorder, monkeypatch, asynchronous, enabled, fail):
    await _pre_call_contract(native_ocr, wire_recorder, monkeypatch, asynchronous, enabled, fail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "asynchronous, control, message",
    [
        (False, "copy-input", "caller identity"),
        (True, "copy-input", "caller identity"),
        (False, "duplicate-prepare", "callback count"),
        (True, "duplicate-prepare", "callback count"),
        (True, "new-task", "callback context/order"),
        (False, "logging-roots", "execution roots"),
        (True, "logging-roots", "execution roots"),
    ],
)
async def test_pre_call_contract_rejects_boundary_mutants(
    native_ocr, wire_recorder, monkeypatch, asynchronous, control, message
):
    with pytest.raises(AssertionError, match=message):
        await _pre_call_contract(native_ocr, wire_recorder, monkeypatch, asynchronous, True, control=control)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_pre_call_contract_allows_root_tuple_reconstruction(native_ocr, wire_recorder, monkeypatch, asynchronous):
    await _pre_call_contract(native_ocr, wire_recorder, monkeypatch, asynchronous, True, control="tuple-only")


class PreCallAbort(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("enabled", [False, True], ids=["python", "native"])
async def test_pre_call_escape_never_sends_or_replays(native_ocr, wire_recorder, monkeypatch, asynchronous, enabled):
    document = {"type": "document_url", "document_url": "https://example.invalid/original.pdf"}
    error = PreCallAbort("stop before POST")
    calls = []
    native_calls = []

    class Abort(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            calls.append(kwargs["additional_args"]["complete_input_dict"]["document"])
            document["document_url"] = "https://example.invalid/aborted.pdf"
            raise error

    def sync_call(boundary):
        native_calls.append("sync")
        return native_ocr.ocr(boundary)

    async def async_call(boundary):
        native_calls.append("async")
        return await native_ocr.aocr(boundary)

    rust_bridge._OCR.override(sync_call)
    rust_bridge._AOCR.override(async_call)
    monkeypatch.setattr(litellm, "input_callback", [Abort()])
    litellm.rust(enabled)
    arguments = dict(model=MODEL, document=document, api_key="sk-test", api_base=wire_recorder.api_base, num_retries=0)
    with pytest.raises(PreCallAbort, match="stop before POST") as caught:
        await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert caught.value is error
    assert len(calls) == 1 and calls[0] is document
    assert native_calls == (["async" if asynchronous else "sync"] if enabled else [])
    assert document["document_url"] == "https://example.invalid/aborted.pdf"
    assert wire_recorder.requests == []


class RetainedDocument(dict):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_ocr_retention_during_post_and_terminal_cleanup(native_ocr, wire_recorder, enabled, outcome):
    retained = []
    references = []
    calls = []
    wire_recorder.release.clear()
    wire_recorder.status = 429 if outcome == "error" else 200

    class Retain(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            body = kwargs["additional_args"]["complete_input_dict"]
            retained.append(body)
            references.append(weakref.ref(body["document"]))
            calls.append("pre_call")

    async def request():
        document = RetainedDocument(type="document_url", document_url="https://example.invalid/original.pdf")
        logging_obj = Logging(
            model="mistral-ocr-latest",
            messages=[],
            stream=False,
            call_type="aocr",
            start_time=datetime.now(),
            litellm_call_id="retention",
            function_id="retention",
            dynamic_input_callbacks=[Retain()],
        )
        arguments = dict(
            model="mistral-ocr-latest",
            document=document,
            optional_params={},
            logging_obj=logging_obj,
            api_key="sk-test",
            api_base=wire_recorder.api_base,
            headers=None,
            provider_config=MistralOCRConfig(),
            litellm_params={},
            custom_llm_provider="mistral",
            timeout=5.0,
        )
        if enabled:
            return await native_ocr.aocr(rust_bridge.OCRBoundary(handler=ocr_main.base_llm_http_handler, **arguments))
        return await ocr_main.base_llm_http_handler.async_ocr(**arguments)

    task = asyncio.create_task(request())
    try:
        assert await asyncio.to_thread(wire_recorder.received.wait, 5), "POST never reached server"
        assert not task.done()
        assert calls == ["pre_call"]
        assert len(references) == 1 and references[0]() is not None
        retained[0]["document"]["document_url"] = "https://example.invalid/after-consumption.pdf"
        retained[0]["document"]["cycle"] = retained[0]
        assert wire_recorder.requests[0]["body"]["document"]["document_url"] == "https://example.invalid/original.pdf"
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "error":
            wire_recorder.release.set()
            with pytest.raises(BaseLLMException) as caught:
                await task
            assert caught.value.status_code == 429
            del caught
        else:
            wire_recorder.release.set()
            response = await task
            assert response.pages[0].markdown == "proof"
    finally:
        wire_recorder.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(wire_recorder.finished.wait, 5), "server did not finish"
    del task
    gc.collect()
    assert references[0]() is retained[0]["document"]
    assert retained[0]["document"]["document_url"] == "https://example.invalid/after-consumption.pdf"
    assert len(wire_recorder.requests) == 1 and calls == ["pre_call"]
    retained.clear()
    await asyncio.sleep(0)
    gc.collect()
    assert references[0]() is None, "execution retained the caller graph after cleanup"


@pytest.mark.parametrize("filename", ["ocr_driver.py", "retained_callback.py"])
def test_native_ocr_cold_cache_reentry(native_ocr, filename):
    script = """
import sys
from litellm.rust_bridge import _native

events = []
compilations = []
error = LookupError('preparation stopped')

class Boundary:
    async def aprepare(self):
        raise error

def invoke():
    pending = _native.aocr(Boundary())
    try:
        pending.send(None)
    except LookupError as caught:
        assert caught is error
        events.append('raised')
    else:
        raise AssertionError('preparation did not raise')
    finally:
        pending.close()
        error.__traceback__ = None

def audit(event, args):
    if event == 'compile' and args[1] == sys.argv[1]:
        compilations.append(args[1])
        if len(compilations) == 1:
            events.append('entered')
            invoke()
            events.append('returned')

sys.addaudithook(audit)
invoke()
invoke()
assert events == ['entered', 'raised', 'returned', 'raised', 'raised'], events
assert len(compilations) == 2, compilations
print('cold reentry passed')
"""
    isolation = ["-I"] if sys.flags.isolated else []
    result = subprocess.run(
        [sys.executable, *isolation, "-c", script, filename], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "cold reentry passed"


@pytest.mark.parametrize("enabled", [False, True], ids=["python", "native"])
def test_sync_pre_call_reentry_without_event_loop(native_ocr, wire_recorder, monkeypatch, enabled):
    context = contextvars.ContextVar("sync-ocr-context", default="caller")
    events = []
    native_calls = []
    document = {"type": "document_url", "document_url": "https://example.invalid/original.pdf"}

    class Reenter(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                loop_running = False
            else:
                loop_running = True
            events.append((context.get(), threading.get_ident(), loop_running))
            if len(events) == 1:
                context.set("nested")
                response = litellm.ocr(
                    model=MODEL,
                    document=dict(document),
                    api_key="sk-test",
                    api_base=wire_recorder.api_base,
                    num_retries=0,
                )
                events.append((response.pages[0].markdown, threading.get_ident(), loop_running))

    def sync_call(boundary):
        native_calls.append(boundary)
        return native_ocr.ocr(boundary)

    rust_bridge._OCR.override(sync_call)
    monkeypatch.setattr(litellm, "input_callback", [Reenter()])
    litellm.rust(enabled)
    response = litellm.ocr(
        model=MODEL,
        document=document,
        api_key="sk-test",
        api_base=wire_recorder.api_base,
        num_retries=0,
    )
    assert response.pages[0].markdown == "proof"
    assert events == [(value, threading.get_ident(), False) for value in ("caller", "nested", "proof")]
    assert context.get() == "nested"
    assert len(native_calls) == (2 if enabled else 0)
    assert len(wire_recorder.requests) == 2


@pytest.mark.parametrize("explicit_close", [False, True], ids=["abandoned", "closed"])
def test_unstarted_native_ocr_driver_releases_cyclic_input(native_ocr, explicit_close):
    document = RetainedDocument(type="document_url", document_url="https://example.invalid/original.pdf")
    reference = weakref.ref(document)
    logger = RecordingLogging()
    boundary = ocr_main._ocr_boundary(build_prepared_request(document=document, logging_obj=logger))
    pending = native_ocr.aocr(boundary)
    document["pending"] = pending
    del boundary, document
    assert reference() is not None
    assert logger.pre_call_kwargs is None
    if explicit_close:
        pending.close()
        del pending
        gc.collect()
    else:
        del pending
        with pytest.warns(RuntimeWarning, match="coroutine .* was never awaited"):
            gc.collect()
    assert reference() is None
    assert logger.pre_call_kwargs is None
