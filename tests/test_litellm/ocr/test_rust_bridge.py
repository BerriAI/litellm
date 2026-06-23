"""Tests for the optional Rust-backed OCR path (``litellm/ocr/rust_bridge.py``).

Covers both the sync ``ocr`` and async ``aocr`` bridge entry points with an
injected fake bridge — the compiled extension is not built in CI.
"""

import asyncio
import importlib

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse

# `litellm/__init__.py` does `from .ocr.main import *`, which binds the `ocr`
# function onto `litellm.ocr` and shadows the submodule, so import the modules
# explicitly via importlib rather than attribute traversal.
ocr_main = importlib.import_module("litellm.ocr.main")
rust_bridge = importlib.import_module("litellm.ocr.rust_bridge")

MODEL = "mistral/mistral-ocr-latest"
DOCUMENT = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}

FAKE_OCR_RESPONSE = {
    "pages": [{"index": 0, "markdown": "hello world"}],
    "model": "mistral-ocr-2505-completion",
    "document_annotation": None,
    "usage_info": {"pages_processed": 1},
    "object": "ocr",
}


class RecordingBridge:
    """A fake bridge exposing sync ``ocr`` and async ``aocr``, recording calls."""

    def __init__(self):
        self.ocr_calls = []
        self.aocr_calls = []

    def _record(self, store, kwargs):
        store.append(kwargs)
        return dict(FAKE_OCR_RESPONSE)

    def ocr(
        self,
        provider,
        model,
        document,
        api_key=None,
        api_base=None,
        extra_headers=None,
        timeout_seconds=None,
        params=None,
    ):
        return self._record(
            self.ocr_calls,
            {
                "provider": provider,
                "model": model,
                "document": document,
                "api_key": api_key,
                "api_base": api_base,
                "extra_headers": extra_headers,
                "timeout_seconds": timeout_seconds,
                "params": params,
            },
        )

    async def aocr(
        self,
        provider,
        model,
        document,
        api_key=None,
        api_base=None,
        extra_headers=None,
        timeout_seconds=None,
        params=None,
    ):
        return self._record(
            self.aocr_calls,
            {
                "provider": provider,
                "model": model,
                "document": document,
                "api_key": api_key,
                "api_base": api_base,
                "extra_headers": extra_headers,
                "timeout_seconds": timeout_seconds,
                "params": params,
            },
        )


class RecordingLogging:
    """A spy standing in for ``LiteLLMLoggingObj`` to capture ``pre_call``."""

    def __init__(self):
        self.pre_call_kwargs = None

    def pre_call(self, *, input, api_key, additional_args):
        self.pre_call_kwargs = {
            "input": input,
            "api_key": api_key,
            "additional_args": additional_args,
        }


class FakeOCRConfig:
    """A stand-in ``BaseOCRConfig`` that echoes the request it would build."""

    def validate_environment(
        self, *, headers, model, api_key, api_base, litellm_params
    ):
        return {**headers, "authorization": f"Bearer {api_key}"}

    def get_complete_url(self, *, api_base, model, optional_params, litellm_params):
        return f"{api_base or 'https://api.mistral.ai/v1'}/ocr"


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    """Keep the global toggle isolated between tests."""
    rust_bridge.use_litellm_rust(False, bridge=None)
    yield
    rust_bridge.use_litellm_rust(False, bridge=None)


@pytest.fixture
def fake_bridge():
    """Enable the Rust path with an injected recording bridge (no native wheel)."""
    bridge = RecordingBridge()
    litellm.use_litellm_rust(True, bridge=bridge)
    return bridge


# --------------------------------------------------------------------------- #
# Toggle + bridge loading
# --------------------------------------------------------------------------- #
def test_use_litellm_rust_toggles_flag():
    assert rust_bridge.rust_ocr_enabled() is False
    litellm.use_litellm_rust()
    assert rust_bridge.rust_ocr_enabled() is True
    litellm.use_litellm_rust(False)
    assert rust_bridge.rust_ocr_enabled() is False


def test_load_rust_bridge_returns_injected_bridge():
    bridge = RecordingBridge()
    litellm.use_litellm_rust(True, bridge=bridge)
    assert rust_bridge.load_rust_bridge() is bridge


def test_toggle_without_bridge_arg_preserves_injection():
    """Routine enable/disable must not clobber a prior injection."""
    bridge = RecordingBridge()
    litellm.use_litellm_rust(True, bridge=bridge)

    litellm.use_litellm_rust(False)
    assert rust_bridge.load_rust_bridge() is bridge
    litellm.use_litellm_rust(True)
    assert rust_bridge.load_rust_bridge() is bridge


def test_explicit_bridge_none_clears_injection():
    bridge = RecordingBridge()
    litellm.use_litellm_rust(True, bridge=bridge)

    litellm.use_litellm_rust(True, bridge=None)
    assert rust_bridge.load_rust_bridge() is None


def test_load_rust_bridge_none_when_extension_absent():
    litellm.use_litellm_rust(True)  # no injection; extension isn't built in CI
    assert rust_bridge.load_rust_bridge() is None


def test_rust_supports_only_known_providers():
    assert rust_bridge.rust_supports("mistral") is True
    assert rust_bridge.rust_supports("azure_ai") is False
    assert rust_bridge.rust_supports("openai") is False


# --------------------------------------------------------------------------- #
# Helper: timeout + sync runner
# --------------------------------------------------------------------------- #
def test_timeout_to_seconds_handles_float_timeout_and_none():
    import httpx

    assert ocr_main._timeout_to_seconds(12.5) == 12.5
    assert ocr_main._timeout_to_seconds(None) is None
    assert ocr_main._timeout_to_seconds(httpx.Timeout(30.0, read=42.0)) == 42.0


def _rust_call(logging_obj, **overrides):
    base = dict(
        logging_obj=logging_obj,
        provider_config=FakeOCRConfig(),
        resolve_api_key=lambda _name: None,
        provider="mistral",
        model="mistral-ocr-latest",
        document=DOCUMENT,
        api_key="sk-test",
        api_base="https://proxy.internal",
        extra_headers=None,
        optional_params={"include_image_base64": True},
        litellm_params={},
        timeout_seconds=12.5,
    )
    base.update(overrides)
    return ocr_main._RustOcrCall(**base)


def test_run_rust_ocr_forwards_args_and_wraps_response():
    bridge = RecordingBridge()
    response = ocr_main._run_rust_ocr(bridge, _rust_call(RecordingLogging()))

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    call = bridge.ocr_calls[0]
    assert call["provider"] == "mistral"
    assert call["model"] == "mistral-ocr-latest"
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    assert call["timeout_seconds"] == 12.5
    assert call["params"] == {"include_image_base64": True}


def test_run_rust_ocr_resolves_key_via_secret_manager_when_missing():
    """No explicit api_key: the resolver (get_secret_str in prod) supplies it."""
    bridge = RecordingBridge()
    ocr_main._run_rust_ocr(
        bridge,
        _rust_call(
            RecordingLogging(),
            api_key=None,
            resolve_api_key=lambda name: (
                "sk-from-vault" if name == "MISTRAL_API_KEY" else None
            ),
        ),
    )
    assert bridge.ocr_calls[0]["api_key"] == "sk-from-vault"


def test_run_rust_ocr_runs_pre_call_logging():
    logging_obj = RecordingLogging()
    ocr_main._run_rust_ocr(
        RecordingBridge(),
        _rust_call(logging_obj, api_base="https://api.mistral.ai/v1"),
    )

    assert logging_obj.pre_call_kwargs is not None
    additional = logging_obj.pre_call_kwargs["additional_args"]
    assert additional["complete_input_dict"]["document"] == DOCUMENT
    assert additional["api_base"] == "https://api.mistral.ai/v1/ocr"
    assert additional["headers"] == {"authorization": "Bearer sk-test"}


def test_arun_rust_ocr_awaits_bridge_aocr():
    bridge = RecordingBridge()
    response = asyncio.run(
        ocr_main._arun_rust_ocr(bridge, _rust_call(RecordingLogging()))
    )
    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    # The async entry point was used, not the sync one.
    assert len(bridge.aocr_calls) == 1
    assert len(bridge.ocr_calls) == 0
    assert bridge.aocr_calls[0]["params"] == {"include_image_base64": True}


# --------------------------------------------------------------------------- #
# Public ocr() / aocr() routing
# --------------------------------------------------------------------------- #
def test_ocr_routes_to_rust_sync(fake_bridge):
    response = litellm.ocr(
        model=MODEL, document=DOCUMENT, api_key="sk-test", include_image_base64=True
    )
    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    assert len(fake_bridge.ocr_calls) == 1
    call = fake_bridge.ocr_calls[0]
    assert call["provider"] == "mistral"
    assert call["model"] == "mistral-ocr-latest"  # provider prefix stripped
    assert call["params"].get("include_image_base64") is True


def test_aocr_routes_to_rust_async(fake_bridge):
    response = asyncio.run(
        litellm.aocr(model=MODEL, document=DOCUMENT, api_key="sk-test")
    )
    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    # Went through the async bridge (no executor thread held during HTTP).
    assert len(fake_bridge.aocr_calls) == 1
    assert len(fake_bridge.ocr_calls) == 0
    assert fake_bridge.aocr_calls[0]["model"] == "mistral-ocr-latest"


def test_ocr_forwards_timeout_to_rust(fake_bridge):
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test", timeout=12.5)
    assert fake_bridge.ocr_calls[0]["timeout_seconds"] == 12.5


def test_ocr_passes_default_request_timeout_to_rust(fake_bridge):
    from litellm.constants import request_timeout

    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")
    assert fake_bridge.ocr_calls[0]["timeout_seconds"] == float(request_timeout)


def test_ocr_does_not_route_to_rust_when_disabled():
    bridge = RecordingBridge()
    litellm.use_litellm_rust(False, bridge=bridge)
    assert rust_bridge.rust_ocr_enabled() is False
    assert bridge.ocr_calls == []


def test_ocr_falls_back_to_python_when_bridge_unavailable(monkeypatch):
    """Rust enabled but no bridge: ocr() must degrade to the Python handler."""
    litellm.use_litellm_rust(True)  # enabled, but load_rust_bridge() returns None

    captured = {}

    def fake_handler_ocr(**kwargs):
        captured["called"] = True
        return OCRResponse(pages=[], model="mistral-ocr-latest", object="ocr")

    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", fake_handler_ocr)
    response = litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured.get("called") is True
    assert isinstance(response, OCRResponse)
