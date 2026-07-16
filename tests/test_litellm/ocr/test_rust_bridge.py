"""Tests for the Rust-backed OCR path (``litellm/ocr/rust_bridge.py``)."""

import importlib
import builtins
import types

import httpx
import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse

# `litellm/__init__.py` does `from .ocr.main import *`, which binds the `ocr`
# function onto `litellm.ocr` and shadows the submodule, so import the modules
# explicitly via importlib rather than attribute traversal.
ocr_main = importlib.import_module("litellm.ocr.main")
rust_bridge = importlib.import_module("litellm.ocr.rust_bridge")
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
    """A fake ``RustOcr`` callable that records the args it was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
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
            }
        )
        return dict(FAKE_OCR_RESPONSE)


class RecordingAsyncBridge:
    """A fake async ``RustAocr`` callable that records the args it was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
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
            }
        )
        return dict(FAKE_OCR_RESPONSE)


class RaisingBridge:
    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        raise RuntimeError("bridge failed")


class RaisingAsyncBridge:
    async def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
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


@pytest.fixture(autouse=True)
def _reset_rust_bridge():
    """Keep the global bridge state isolated between tests."""
    rust_bridge._set_rust_ocr_bridge(ocr=None, aocr=None)
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL
    yield
    rust_bridge._set_rust_ocr_bridge(ocr=None, aocr=None)
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL


@pytest.fixture
def fake_bridge():
    """Inject a recording bridge."""
    bridge = RecordingBridge()
    rust_bridge._set_rust_ocr_bridge(ocr=bridge)
    return bridge


@pytest.fixture
def fake_async_bridge():
    """Inject an async recording bridge."""
    bridge = RecordingAsyncBridge()
    rust_bridge._set_rust_ocr_bridge(aocr=bridge)
    return bridge


def test_load_rust_ocr_returns_injected_impl():
    bridge = RecordingBridge()
    rust_bridge._set_rust_ocr_bridge(ocr=bridge)
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


def test_native_bridge_available_reflects_loader(monkeypatch):
    fake_module = types.ModuleType("litellm.rust_bridge._native")
    monkeypatch.setattr(rust_bridge_loader, "get_native_bridge", lambda: fake_module)

    assert rust_bridge_loader.native_bridge_available() is True


def test_load_rust_aocr_returns_injected_impl():
    bridge = RecordingAsyncBridge()
    rust_bridge._set_rust_ocr_bridge(aocr=bridge)
    assert rust_bridge.load_rust_aocr() is bridge


def test_bridge_injection_preserves_unspecified_impl():
    bridge = RecordingBridge()
    async_bridge = RecordingAsyncBridge()
    rust_bridge._set_rust_ocr_bridge(ocr=bridge, aocr=async_bridge)

    rust_bridge._set_rust_ocr_bridge()
    assert rust_bridge.load_rust_ocr() is bridge
    assert rust_bridge.load_rust_aocr() is async_bridge


def test_explicit_ocr_none_clears_injected_impl(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("litellm.rust_bridge"),
        "get_native_bridge",
        lambda: None,
    )
    bridge = RecordingBridge()
    async_bridge = RecordingAsyncBridge()
    rust_bridge._set_rust_ocr_bridge(ocr=bridge, aocr=async_bridge)

    rust_bridge._set_rust_ocr_bridge(ocr=None, aocr=None)
    assert rust_bridge.load_rust_ocr() is None
    assert rust_bridge.load_rust_aocr() is None


def test_load_rust_ocr_none_when_extension_absent(monkeypatch):
    """With no injected impl and no compiled wheel, the loader returns None so the
    caller degrades to the Python path instead of raising ImportError."""
    monkeypatch.setattr(
        importlib.import_module("litellm.rust_bridge"),
        "get_native_bridge",
        lambda: None,
    )
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
        importlib.import_module("litellm.rust_bridge"),
        "get_native_bridge",
        lambda: fake_module,
    )

    assert rust_bridge.load_rust_ocr() is fake_module.ocr
    assert rust_bridge.load_rust_aocr() is fake_module.aocr


def test_timeout_to_seconds_handles_float_timeout_and_none():
    assert ocr_main._timeout_to_seconds(12.5) == 12.5
    assert ocr_main._timeout_to_seconds(None) is None
    assert ocr_main._timeout_to_seconds(httpx.Timeout(30.0, read=42.0)) == 42.0


def test_run_rust_ocr_forwards_args_and_wraps_response():
    bridge = RecordingBridge()
    logging_obj = RecordingLogging()

    response = ocr_main._run_rust_ocr(
        rust_ocr=bridge,
        model="mistral-ocr-latest",
        document=DOCUMENT,
        api_key="sk-test",
        api_base="https://proxy.internal",
        custom_llm_provider="mistral",
        extra_headers={"x-trace-id": "trace-1"},
        optional_params={"include_image_base64": True},
        timeout=12.5,
        litellm_logging_obj=logging_obj,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    call = bridge.calls[0]
    assert call == {
        "model": "mistral-ocr-latest",
        "document": DOCUMENT,
        "api_key": "sk-test",
        "api_base": "https://proxy.internal",
        "custom_llm_provider": "mistral",
        "extra_headers": {"x-trace-id": "trace-1"},
        "optional_params": {"include_image_base64": True},
        "timeout_seconds": 12.5,
    }

def test_run_rust_ocr_runs_pre_call_logging():
    """The Rust shortcut must run pre_call so callbacks and spend tracking fire."""
    logging_obj = RecordingLogging()

    ocr_main._run_rust_ocr(
        rust_ocr=RecordingBridge(),
        model="mistral-ocr-latest",
        document=DOCUMENT,
        api_key="sk-test",
        api_base="https://api.mistral.ai/v1",
        custom_llm_provider="mistral",
        extra_headers={"x-trace-id": "trace-1"},
        optional_params={"include_image_base64": True},
        timeout=12.5,
        litellm_logging_obj=logging_obj,
    )

    assert logging_obj.pre_call_kwargs is not None
    assert logging_obj.pre_call_kwargs["input"] == "OCR document processing"
    additional_args = logging_obj.pre_call_kwargs["additional_args"]
    complete_input = additional_args["complete_input_dict"]
    assert complete_input["document"] == DOCUMENT
    assert complete_input["include_image_base64"] is True
    assert additional_args["api_base"] == "https://api.mistral.ai/v1"
    assert additional_args["headers"] == {"x-trace-id": "trace-1"}


def test_ocr_routes_to_rust_by_default(fake_bridge):
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
    # Provider prefix is stripped before reaching the bridge.
    assert call["model"] == "mistral-ocr-latest"
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    assert call["custom_llm_provider"] == "mistral"
    assert call["extra_headers"] == {"x-trace-id": "trace-1"}
    # Raw OCR params ride along in optional_params; Rust filters to supported keys.
    assert call["optional_params"].get("include_image_base64") is True


def test_ocr_filters_internal_litellm_params_before_rust(fake_bridge):
    litellm.ocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
        include_image_base64=True,
        original_generic_function=lambda: None,
        litellm_metadata={"trace": "internal"},
    )

    assert fake_bridge.calls[0]["optional_params"] == {"include_image_base64": True}


def test_ocr_forwards_public_id_but_drops_internal_litellm_params(fake_bridge):
    litellm.ocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
        id="ocr-req-9",
        pages=[0, 1],
        include_image_base64=True,
        table_format="html",
        metadata={"trace": "internal"},
        litellm_metadata={"trace": "internal"},
        num_retries=3,
        original_generic_function=lambda: None,
    )

    optional_params = fake_bridge.calls[0]["optional_params"]
    assert optional_params["id"] == "ocr-req-9"
    assert optional_params["pages"] == [0, 1]
    assert optional_params["include_image_base64"] is True
    assert optional_params["table_format"] == "html"
    for internal in (
        "metadata",
        "litellm_metadata",
        "num_retries",
        "original_generic_function",
    ):
        assert internal not in optional_params


def test_ocr_routes_azure_ai_to_rust_by_default(fake_bridge):
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


def test_ocr_exception_type_uses_resolved_provider_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_exception_type(**kwargs: object) -> CapturedException:
        captured.update(kwargs)
        return CapturedException("wrapped")

    monkeypatch.setattr(ocr_main.litellm, "exception_type", fake_exception_type)
    rust_bridge._set_rust_ocr_bridge(ocr=RaisingBridge())

    with pytest.raises(CapturedException):
        litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured["model"] == "mistral-ocr-latest"
    assert captured["custom_llm_provider"] == "mistral"


@pytest.mark.asyncio
async def test_aocr_routes_to_async_rust_by_default(fake_async_bridge):
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
    rust_bridge._set_rust_ocr_bridge(aocr=RaisingAsyncBridge())

    with pytest.raises(CapturedException):
        await litellm.aocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured["model"] == "mistral-ocr-latest"
    assert captured["custom_llm_provider"] == "mistral"


def test_ocr_forwards_timeout_to_rust(fake_bridge):
    """Caller-supplied timeout must flow into the Rust bridge so the fixed 600s
    client ceiling doesn't silently override shorter deadlines."""
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test", timeout=12.5)

    assert fake_bridge.calls[0]["timeout_seconds"] == 12.5


def test_ocr_passes_default_request_timeout_to_rust(fake_bridge):
    """When no explicit timeout is given, the library default (request_timeout)
    must still be forwarded so the Rust path matches the Python path's deadline."""
    from litellm.constants import request_timeout

    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert fake_bridge.calls[0]["timeout_seconds"] == float(request_timeout)


def test_ocr_requires_rust_bridge_when_unavailable(monkeypatch):
    monkeypatch.setattr(ocr_main, "load_rust_ocr", lambda: None)

    with pytest.raises(Exception) as exc_info:
        litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert "Rust OCR bridge is required" in str(exc_info.value)
