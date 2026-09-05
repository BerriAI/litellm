"""Tests for the optional Rust-backed OCR path."""

import importlib
import os
import subprocess
from pathlib import Path
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge.callbacks import OneShotCallbackHandle
from litellm.rust_bridge.request import NativeRequestContext, NativeRequestOptions, PreparedNativeCall
from litellm.rust_bridge.timeouts import timeout_to_seconds

# `litellm/__init__.py` does `from .ocr.main import *`, which binds the `ocr`
# function onto `litellm.ocr` and shadows the submodule, so import the modules
# explicitly via importlib rather than attribute traversal.
ocr_main = importlib.import_module("litellm.ocr.main")
rust_bridge = importlib.import_module("litellm.rust_bridge.ocr")
rust_bridge_loader = importlib.import_module("litellm.rust_bridge.loader")

MODEL = "mistral/mistral-ocr-latest"
DOCUMENT: dict[str, object] = {
    "type": "document_url",
    "document_url": "https://example.com/doc.pdf",
}


def test_installed_wheel_ocr_callback_parity() -> None:
    wheel_python: Final = os.environ.get("LITELLM_OCR_WHEEL_PYTHON")
    if wheel_python is None:
        pytest.skip(
            "set LITELLM_OCR_WHEEL_PYTHON to the reviewed wheel's interpreter; release-wheel CI requires this lane"
        )
    script: Final = Path(__file__).resolve().parents[1] / "rust_bridge" / "sdk_callback_wheel_test.py"
    completed: Final = subprocess.run((wheel_python, str(script)), check=False, timeout=240)
    assert completed.returncode == 0


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
    """A fake ``RustOcr`` callable that records the args it was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.requests: list[rust_bridge.NativeOCRRequest] = []

    def __call__(
        self,
        request: rust_bridge.NativeOCRRequest,
        *,
        context: NativeRequestContext,
        callback_adapter: OneShotCallbackHandle | None = None,
    ) -> dict[str, object]:
        model: Final = request.model
        document: Final = request.document
        api_key: Final = request.options.api_key
        api_base: Final = request.options.api_base
        custom_llm_provider: Final = request.options.custom_llm_provider
        extra_headers: Final = request.options.extra_headers
        optional_params: Final = request.optional_params
        timeout_seconds: Final = request.options.timeout_seconds
        litellm_call_id: Final = context.litellm_call_id
        self.requests.append(request)
        self.calls.append(
            {
                "model": model,
                "document": document,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_seconds,
                "litellm_call_id": litellm_call_id,
            }
        )
        return dict(FAKE_OCR_RESPONSE)


class RecordingAsyncBridge:
    """A fake async ``RustAocr`` callable that records the args it was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.requests: list[rust_bridge.NativeOCRRequest] = []

    async def __call__(
        self,
        request: rust_bridge.NativeOCRRequest,
        *,
        context: NativeRequestContext,
        callback_adapter: OneShotCallbackHandle | None = None,
    ) -> dict[str, object]:
        litellm_call_id: Final = context.litellm_call_id
        self.requests.append(request)
        model: Final = request.model
        document: Final = request.document
        api_key: Final = request.options.api_key
        api_base: Final = request.options.api_base
        custom_llm_provider: Final = request.options.custom_llm_provider
        extra_headers: Final = request.options.extra_headers
        optional_params: Final = request.optional_params
        timeout_seconds: Final = request.options.timeout_seconds
        self.calls.append(
            {
                "model": model,
                "document": document,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_seconds,
                "litellm_call_id": litellm_call_id,
            }
        )
        return dict(FAKE_OCR_RESPONSE)


class RaisingBridge:
    def __call__(
        self,
        request: rust_bridge.NativeOCRRequest,
        *,
        context: NativeRequestContext,
        callback_adapter: OneShotCallbackHandle | None = None,
    ) -> dict[str, object]:
        raise RuntimeError("bridge failed")


async def _async_none() -> None:
    return None


class RaisingAsyncBridge:
    async def __call__(
        self,
        request: rust_bridge.NativeOCRRequest,
        *,
        context: NativeRequestContext,
        callback_adapter: OneShotCallbackHandle | None = None,
    ) -> dict[str, object]:
        raise RuntimeError("bridge failed")


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    """Keep the global toggle isolated between tests."""
    rust_bridge.set_rust_ocr(ocr=None, aocr=None, decline=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL
    rust_bridge.set_rust_ocr(
        decline=lambda model, custom_llm_provider, **features: (
            "unsupported feature"
            if any(features.get(key) for key in ("stream", "has_agentic_hook", "has_custom_client"))
            or features.get("request_format") == "native"
            else None
        )
    )
    yield
    rust_bridge.set_rust_ocr(ocr=None, aocr=None, decline=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL


@pytest.fixture
def fake_bridge():
    """Enable the Rust path with an injected recording bridge (no native wheel)."""
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge.set_rust_ocr(ocr=bridge)
    return bridge


@pytest.fixture
def fake_async_bridge():
    """Enable the async Rust path with an injected recording bridge."""
    bridge = RecordingAsyncBridge()
    litellm.rust(True)
    rust_bridge.set_rust_ocr(aocr=bridge)
    return bridge


def test_bridge_wrapper_forwards_prepared_args_and_wraps_response():
    bridge = RecordingBridge()

    litellm.rust(True)

    rust_bridge.set_rust_ocr(ocr=bridge)
    response = rust_bridge.dispatch_ocr(
        prepare=lambda: PreparedNativeCall(
            request=rust_bridge.NativeOCRRequest(
                model="mistral-ocr-latest",
                document=DOCUMENT,
                optional_params={"include_image_base64": True, "pages": [0]},
                options=NativeRequestOptions(
                    api_key="sk-test",
                    api_base="https://proxy.internal",
                    custom_llm_provider="mistral",
                    extra_headers={"Authorization": "Bearer sk-test", "x-trace-id": "trace-1"},
                    timeout_seconds=timeout_to_seconds(12.5),
                ),
            )
        ),
        fallback=lambda: None,
        adapt=lambda value: value,
        model="mistral-ocr-latest",
        provider="mistral",
        eligible=True,
    )

    assert response == FAKE_OCR_RESPONSE
    call = bridge.calls[0]
    assert call == {
        "model": "mistral-ocr-latest",
        "document": DOCUMENT,
        "api_key": "sk-test",
        "api_base": "https://proxy.internal",
        "custom_llm_provider": "mistral",
        "extra_headers": {
            "Authorization": "Bearer sk-test",
            "x-trace-id": "trace-1",
        },
        "optional_params": {"include_image_base64": True, "pages": [0]},
        "timeout_seconds": 12.5,
        "litellm_call_id": None,
    }


@pytest.mark.asyncio
async def test_bridge_wrapper_forwards_prepared_async_args_and_wraps_response():
    bridge = RecordingAsyncBridge()

    litellm.rust(True)

    rust_bridge.set_rust_ocr(aocr=bridge)
    response = await rust_bridge.adispatch_ocr(
        prepare=lambda: PreparedNativeCall(
            request=rust_bridge.NativeOCRRequest(
                model="mistral-ocr-maas",
                document=DOCUMENT,
                optional_params={"vertex_project": "project-1"},
                options=NativeRequestOptions(
                    api_key=None,
                    api_base=None,
                    custom_llm_provider="vertex_ai",
                    extra_headers=None,
                    timeout_seconds=timeout_to_seconds(httpx.Timeout(30.0, read=42.0)),
                ),
            )
        ),
        fallback=_async_none,
        adapt=lambda value: value,
        model="mistral-ocr-maas",
        provider="vertex_ai",
        eligible=True,
    )

    assert response == FAKE_OCR_RESPONSE
    assert bridge.calls[0] == {
        "model": "mistral-ocr-maas",
        "document": DOCUMENT,
        "api_key": None,
        "api_base": None,
        "custom_llm_provider": "vertex_ai",
        "extra_headers": None,
        "optional_params": {"vertex_project": "project-1"},
        "timeout_seconds": 42.0,
        "litellm_call_id": None,
    }


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
    assert call["model"] == MODEL
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    assert call["custom_llm_provider"] is None
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
    assert fake_bridge.calls[0]["model"] == "azure_ai/pixtral-12b-2409"
    assert fake_bridge.calls[0]["custom_llm_provider"] is None


def test_file_document_uses_python_without_entering_native(fake_bridge, monkeypatch):
    handler = Mock(return_value=OCRResponse(pages=[], model="mistral-ocr-latest", object="ocr"))
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", handler)
    litellm.ocr(
        model=MODEL, document={"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"}, api_key="test-key"
    )
    assert not fake_bridge.calls
    document = handler.call_args.kwargs["document"]
    assert document["document_url"].startswith("data:application/pdf;base64,")


def test_ocr_exception_type_preserves_raw_request_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_exception_type(**kwargs: object) -> CapturedException:
        captured.update(kwargs)
        return CapturedException("wrapped")

    monkeypatch.setattr(ocr_main.litellm, "exception_type", fake_exception_type)
    litellm.rust(True)
    rust_bridge.set_rust_ocr(ocr=RaisingBridge())

    with pytest.raises(CapturedException):
        litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured["model"] == MODEL
    assert captured["custom_llm_provider"] is None


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
    assert call["model"] == MODEL
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    assert call["custom_llm_provider"] is None
    assert call["extra_headers"] == {"x-trace-id": "trace-1"}
    assert call["optional_params"].get("include_image_base64") is True


@pytest.mark.asyncio
async def test_aocr_exception_type_preserves_raw_request_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_exception_type(**kwargs: object) -> CapturedException:
        captured.update(kwargs)
        return CapturedException("wrapped")

    monkeypatch.setattr(ocr_main.litellm, "exception_type", fake_exception_type)
    litellm.rust(True)
    rust_bridge.set_rust_ocr(aocr=RaisingAsyncBridge())

    with pytest.raises(CapturedException):
        await litellm.aocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured["model"] == MODEL
    assert captured["custom_llm_provider"] is None


def test_ocr_forwards_timeout_to_rust(fake_bridge):
    """Caller-supplied timeout must flow into the Rust bridge so the fixed 600s
    client ceiling doesn't silently override shorter deadlines."""
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test", timeout=12.5)

    assert fake_bridge.calls[0]["timeout_seconds"] == 12.5


def test_ocr_passes_default_request_timeout_to_rust(fake_bridge):
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    from litellm.constants import request_timeout

    assert fake_bridge.calls[0]["timeout_seconds"] == float(request_timeout)


def test_ocr_disabled_never_loads_or_prepares_rust(monkeypatch: pytest.MonkeyPatch):
    def fail() -> None:
        pytest.fail("disabled OCR must not load or prepare Rust")

    python_ocr: Final = Mock(return_value=OCRResponse(pages=[], model="mistral-ocr-latest", object="ocr"))
    litellm.rust(False)
    monkeypatch.setattr(bindings, "get_native_bridge", fail)
    monkeypatch.setattr(rust_bridge, "prepare_call", fail)
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", python_ocr)

    response: Final = litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert isinstance(response, OCRResponse)
    python_ocr.assert_called_once()


def test_ocr_falls_back_to_python_when_bridge_unavailable(monkeypatch):
    """Rust enabled but no bridge available (no injected impl, no compiled wheel):
    ocr() must degrade to the Python HTTP handler instead of raising."""
    rust_bridge._OCR.sync.override(None)
    litellm.rust(True)

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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_handoff_does_not_prepare_provider_in_python(monkeypatch, asynchronous):
    def unexpected(*args, **kwargs):
        pytest.fail("native handoff must not run Python OCR preparation or secret resolution")

    monkeypatch.setattr(ocr_main, "_prepare_ocr_request", unexpected)
    secret_manager = importlib.import_module("litellm.secret_managers.main")
    monkeypatch.setattr(secret_manager, "get_secret_str", unexpected)
    bridge = RecordingAsyncBridge() if asynchronous else RecordingBridge()
    litellm.rust(True)
    if asynchronous:
        rust_bridge.set_rust_ocr(aocr=bridge)
        await litellm.aocr(model=MODEL, document=DOCUMENT, pages=[1])
    else:
        rust_bridge.set_rust_ocr(ocr=bridge)
        litellm.ocr(model=MODEL, document=DOCUMENT, pages=[1])
    request = bridge.requests[0]
    assert request.options.api_key is None
    assert request.options.extra_headers is None
    assert request.optional_params["pages"] == [1]
    assert request.model == MODEL


def test_auth_callable_is_forwarded_without_invocation(monkeypatch):
    def token_provider():
        pytest.fail("Python handoff must not acquire credentials")

    monkeypatch.setattr(litellm, "api_key", "sdk-default")
    prepared = rust_bridge.prepare_call(
        model=MODEL,
        document=DOCUMENT,
        api_key="explicit",
        api_base=None,
        timeout=None,
        custom_llm_provider=None,
        extra_headers=None,
        kwargs={"azure_ad_token_provider": token_provider, "azure_tenant_id": "tenant"},
        asynchronous=False,
    )
    assert prepared.auth_provider is token_provider
    assert prepared.request.options.api_key == "explicit"
    assert prepared.request.options.provider_connection["sdk_api_key"] == "sdk-default"
    assert prepared.request.options.provider_connection["azure_tenant_id"] == "tenant"


def test_custom_secret_resolver_keeps_python_path(fake_bridge, monkeypatch):
    handler = Mock(return_value=OCRResponse(pages=[], model="mistral-ocr-latest", object="ocr"))
    monkeypatch.setattr(litellm, "secret_manager_client", object())
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", handler)
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="explicit")
    assert not fake_bridge.calls
    handler.assert_called_once()


def test_process_disabled_keeps_python_path(fake_bridge, monkeypatch):
    handler = Mock(return_value=OCRResponse(pages=[], model="mistral-ocr-latest", object="ocr"))
    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", handler)
    litellm.rust(False)
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="explicit")
    assert not fake_bridge.calls
    handler.assert_called_once()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["missing_preflight", "declined", "error", "malformed", "cancelled"])
@pytest.mark.asyncio
async def test_public_ocr_falls_back_only_on_decline(monkeypatch, asynchronous, outcome):
    import asyncio
    from types import SimpleNamespace

    class Declined(Exception):
        pass

    class Upstream(Exception):
        pass

    monkeypatch.setattr(
        bindings, "get_native_bridge", lambda: SimpleNamespace(RustBridgeDeclined=Declined, RustUpstreamError=Upstream)
    )
    calls = []

    def python(**kwargs):
        calls.append(kwargs)
        return OCRResponse(pages=[], model="python", object="ocr")

    monkeypatch.setattr(ocr_main, "base_llm_http_handler", SimpleNamespace(ocr=python))

    def native(request, *, context, callback_adapter=None):
        if outcome == "declined":
            raise Declined("unsupported document")
        if outcome == "error":
            raise RuntimeError("provider failed")
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        return {"pages": "invalid"}

    async def anative(request, *, context, callback_adapter=None):
        return native(request, context=context)

    litellm.rust(True)
    rust_bridge.set_rust_ocr(ocr=native, aocr=anative)
    if outcome == "missing_preflight":
        rust_bridge._PREFLIGHT.override(None)

    async def run():
        kwargs = {"model": MODEL, "document": DOCUMENT, "api_key": "key", "num_retries": 0}
        return await litellm.aocr(**kwargs) if asynchronous else litellm.ocr(**kwargs)

    if outcome in {"declined", "missing_preflight"}:
        assert (await run()).model == "python"
        assert len(calls) == 1
    else:
        with pytest.raises(
            asyncio.CancelledError if outcome == "cancelled" else (RuntimeError, ValueError, litellm.APIConnectionError)
        ):
            await run()
        assert calls == []
