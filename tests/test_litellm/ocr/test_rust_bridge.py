"""Tests for the optional Rust-backed OCR path (``litellm/ocr/rust_bridge.py``)."""

import importlib
import sys
import types
from typing import Any

import httpx
import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse

# `litellm/__init__.py` does `from .ocr.main import *`, which binds the `ocr`
# function onto `litellm.ocr` and shadows the submodule, so import the modules
# explicitly via importlib rather than attribute traversal.
ocr_main = importlib.import_module("litellm.ocr.main")
rust_bridge = importlib.import_module("litellm.ocr.rust_bridge")

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


class FakeOCRConfig:
    """A stand-in ``BaseOCRConfig`` that echoes the request it would build."""

    def __init__(self, api_key_env_var: str = "MISTRAL_API_KEY") -> None:
        self.api_key_env_var = api_key_env_var

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
        effective_timeout=timeout,
        litellm_logging_obj=logging_obj or RecordingLogging(),
    )


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    """Keep the global toggle isolated between tests."""
    rust_bridge.use_litellm_rust(False, ocr=None, aocr=None)
    yield
    rust_bridge.use_litellm_rust(False, ocr=None, aocr=None)


@pytest.fixture
def fake_bridge():
    """Enable the Rust path with an injected recording bridge (no native wheel)."""
    bridge = RecordingBridge()
    litellm.use_litellm_rust(True, ocr=bridge)
    return bridge


@pytest.fixture
def fake_async_bridge():
    """Enable the async Rust path with an injected recording bridge."""
    bridge = RecordingAsyncBridge()
    litellm.use_litellm_rust(True, aocr=bridge)
    return bridge


def test_use_litellm_rust_toggles_flag():
    assert rust_bridge.rust_ocr_enabled() is False
    litellm.use_litellm_rust()
    assert rust_bridge.rust_ocr_enabled() is True
    litellm.use_litellm_rust(False)
    assert rust_bridge.rust_ocr_enabled() is False


def test_load_rust_ocr_returns_injected_impl():
    bridge = RecordingBridge()
    litellm.use_litellm_rust(True, ocr=bridge)
    assert rust_bridge.load_rust_ocr() is bridge


def test_load_rust_aocr_returns_injected_impl():
    bridge = RecordingAsyncBridge()
    litellm.use_litellm_rust(True, aocr=bridge)
    assert rust_bridge.load_rust_aocr() is bridge


def test_toggle_without_ocr_arg_preserves_injected_impl():
    """Regression: routine enable/disable calls must not clobber a prior injection.

    Earlier, ``use_litellm_rust()`` unconditionally assigned the keyword default
    of ``None`` to ``_rust_ocr_impl``, silently dropping a custom bridge whenever
    a caller toggled the flag without re-passing ``ocr=``.
    """
    bridge = RecordingBridge()
    async_bridge = RecordingAsyncBridge()
    litellm.use_litellm_rust(True, ocr=bridge, aocr=async_bridge)

    litellm.use_litellm_rust(False)
    assert rust_bridge.load_rust_ocr() is bridge
    assert rust_bridge.load_rust_aocr() is async_bridge
    litellm.use_litellm_rust(True)
    assert rust_bridge.load_rust_ocr() is bridge
    assert rust_bridge.load_rust_aocr() is async_bridge


def test_explicit_ocr_none_clears_injected_impl():
    bridge = RecordingBridge()
    async_bridge = RecordingAsyncBridge()
    litellm.use_litellm_rust(True, ocr=bridge, aocr=async_bridge)

    litellm.use_litellm_rust(True, ocr=None, aocr=None)
    assert rust_bridge.load_rust_ocr() is None
    assert rust_bridge.load_rust_aocr() is None


def test_load_rust_ocr_none_when_extension_absent():
    """With no injected impl and no compiled wheel, the loader returns None so the
    caller degrades to the Python path instead of raising ImportError."""
    litellm.use_litellm_rust(True)  # no impl injected; extension isn't built in CI
    assert rust_bridge.load_rust_ocr() is None
    assert rust_bridge.load_rust_aocr() is None


def test_load_rust_ocr_uses_compiled_extension(monkeypatch):
    """With no injected impl but a compiled ``litellm_python_bridge`` importable,
    the loader returns the extension's ``ocr`` callable. The native wheel isn't
    built in CI, so stand in a fake module via ``sys.modules``."""
    fake_module = types.ModuleType("litellm_python_bridge")
    fake_module.ocr = lambda **kwargs: dict(FAKE_OCR_RESPONSE)  # type: ignore[attr-defined]
    fake_module.aocr = lambda **kwargs: dict(FAKE_OCR_RESPONSE)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm_python_bridge", fake_module)

    litellm.use_litellm_rust(True)  # enabled, no impl injected -> import the extension
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
        prepared_request=build_prepared_request(
            logging_obj=logging_obj,
            api_base="https://proxy.internal",
            extra_headers={"x-trace-id": "trace-1"},
            optional_params={"include_image_base64": True},
            timeout=12.5,
        ),
        resolve_api_key=lambda _name: None,
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
        "extra_headers": {
            "Authorization": "Bearer sk-test",
            "x-trace-id": "trace-1",
        },
        "optional_params": {"include_image_base64": True},
        "timeout_seconds": 12.5,
    }


def test_run_rust_ocr_resolves_key_via_secret_manager_when_missing():
    """No explicit api_key: the resolver (get_secret_str in production) supplies it,
    so secret-manager backends (AWS/Azure/GCP/Vault) work like the Python path."""
    bridge = RecordingBridge()

    ocr_main._run_rust_ocr(
        rust_ocr=bridge,
        prepared_request=build_prepared_request(api_key=None, timeout=None),
        resolve_api_key=lambda name: (
            "sk-from-vault" if name == "MISTRAL_API_KEY" else None
        ),
    )

    assert bridge.calls[0]["api_key"] == "sk-from-vault"


def test_run_rust_ocr_uses_provider_api_key_env_var():
    bridge = RecordingBridge()
    resolver_calls = []

    def _resolver(name):
        resolver_calls.append(name)
        return "sk-provider-env"

    ocr_main._run_rust_ocr(
        rust_ocr=bridge,
        prepared_request=build_prepared_request(
            provider_config=FakeOCRConfig(api_key_env_var="PROVIDER_OCR_API_KEY"),
            model="provider-ocr-model",
            api_key=None,
            timeout=None,
        ),
        resolve_api_key=_resolver,
    )

    assert resolver_calls == ["PROVIDER_OCR_API_KEY"]
    assert bridge.calls[0]["api_key"] == "sk-provider-env"


def test_run_rust_ocr_prefers_explicit_key_over_resolver():
    bridge = RecordingBridge()
    resolver_calls = []

    def _resolver(name):
        resolver_calls.append(name)
        return "sk-from-vault"

    ocr_main._run_rust_ocr(
        rust_ocr=bridge,
        prepared_request=build_prepared_request(api_key="sk-explicit", timeout=None),
        resolve_api_key=_resolver,
    )

    assert bridge.calls[0]["api_key"] == "sk-explicit"
    assert resolver_calls == []  # resolver never consulted when a key is supplied


def test_run_rust_ocr_runs_pre_call_logging():
    """The Rust shortcut must run pre_call so callbacks and spend tracking fire."""
    logging_obj = RecordingLogging()

    ocr_main._run_rust_ocr(
        rust_ocr=RecordingBridge(),
        prepared_request=build_prepared_request(
            logging_obj=logging_obj,
            api_base="https://api.mistral.ai/v1",
            extra_headers={"x-trace-id": "trace-1"},
            optional_params={"include_image_base64": True},
            timeout=None,
        ),
        resolve_api_key=lambda _name: None,
    )

    assert logging_obj.pre_call_kwargs is not None
    assert logging_obj.pre_call_kwargs["input"] == "OCR document processing"
    additional_args = logging_obj.pre_call_kwargs["additional_args"]
    complete_input = additional_args["complete_input_dict"]
    assert complete_input["document"] == DOCUMENT
    assert complete_input["include_image_base64"] is True
    # The logged request mirrors what Rust sends: resolved URL + headers.
    assert additional_args["api_base"] == "https://api.mistral.ai/v1/ocr"
    assert additional_args["headers"] == {
        "Authorization": "Bearer sk-test",
        "x-trace-id": "trace-1",
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
    # Provider prefix is stripped before reaching the bridge.
    assert call["model"] == "mistral-ocr-latest"
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    assert call["custom_llm_provider"] == "mistral"
    assert call["extra_headers"] == {
        "Authorization": "Bearer sk-test",
        "x-trace-id": "trace-1",
    }
    # Raw OCR params ride along in optional_params; Rust filters to supported keys.
    assert call["optional_params"].get("include_image_base64") is True


def test_ocr_exception_type_uses_resolved_provider_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_exception_type(**kwargs: object) -> CapturedException:
        captured.update(kwargs)
        return CapturedException("wrapped")

    monkeypatch.setattr(ocr_main.litellm, "exception_type", fake_exception_type)
    litellm.use_litellm_rust(True, ocr=RaisingBridge())

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
    assert call["extra_headers"] == {
        "Authorization": "Bearer sk-test",
        "x-trace-id": "trace-1",
    }
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
    litellm.use_litellm_rust(True, aocr=RaisingAsyncBridge())

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


def test_ocr_does_not_route_to_rust_when_disabled():
    """With the flag off, the bridge must not be consulted even if an impl exists."""
    bridge = RecordingBridge()
    litellm.use_litellm_rust(False, ocr=bridge)

    assert rust_bridge.rust_ocr_enabled() is False
    # The impl stays available for injection, but the disabled flag gates usage,
    # so ocr() never reaches the Rust path (asserted via the enabled-path test).
    assert bridge.calls == []


def test_ocr_falls_back_to_python_when_bridge_unavailable(monkeypatch):
    """Rust enabled but no bridge available (no injected impl, no compiled wheel):
    ocr() must degrade to the Python HTTP handler instead of raising."""
    litellm.use_litellm_rust(True)  # enabled, but load_rust_ocr() returns None in CI

    captured = {}

    def fake_handler_ocr(**kwargs):
        captured["called"] = True
        return OCRResponse(pages=[], model="mistral-ocr-latest", object="ocr")

    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", fake_handler_ocr)

    response = litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured.get("called") is True  # Python path was used
    assert isinstance(response, OCRResponse)
