"""Tests for the optional Rust-backed OCR path (``litellm/ocr/rust_bridge.py``)."""

import importlib
import sys
import types

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse

# `litellm/__init__.py` does `from .ocr.main import *`, which binds the `ocr`
# function onto `litellm.ocr` and shadows the submodule — so import the modules
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


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    """Keep the global toggle isolated between tests."""
    rust_bridge.use_litellm_rust(False)
    yield
    rust_bridge.use_litellm_rust(False)


@pytest.fixture
def fake_bridge(monkeypatch):
    """Install a fake compiled ``litellm_python_bridge`` module and record calls."""
    calls = []

    def _ocr(model, document, api_key, api_base, optional_params, timeout_seconds=None):
        calls.append(
            {
                "model": model,
                "document": document,
                "api_key": api_key,
                "api_base": api_base,
                "optional_params": optional_params,
                "timeout_seconds": timeout_seconds,
            }
        )
        return dict(FAKE_OCR_RESPONSE)

    module = types.ModuleType("litellm_python_bridge")
    module.ocr = _ocr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm_python_bridge", module)
    return calls


def test_use_litellm_rust_toggles_flag():
    assert rust_bridge.rust_ocr_enabled() is False
    litellm.use_litellm_rust()
    assert rust_bridge.rust_ocr_enabled() is True
    litellm.use_litellm_rust(False)
    assert rust_bridge.rust_ocr_enabled() is False


def test_rust_ocr_returns_bridge_dict(fake_bridge):
    response = rust_bridge.rust_ocr(
        model="mistral-ocr-latest",
        document=DOCUMENT,
        api_key="sk-test",
        api_base=None,
        optional_params={"include_image_base64": True},
    )

    # rust_ocr returns the raw dict; main.py wraps it into an OCRResponse.
    assert isinstance(response, dict)
    assert response["pages"][0]["markdown"] == "hello world"
    assert response["model"] == "mistral-ocr-2505-completion"
    assert fake_bridge[0]["model"] == "mistral-ocr-latest"
    assert fake_bridge[0]["optional_params"] == {"include_image_base64": True}


def test_ocr_routes_to_rust_when_enabled(fake_bridge):
    litellm.use_litellm_rust()

    response = litellm.ocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
        include_image_base64=True,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    assert len(fake_bridge) == 1
    call = fake_bridge[0]
    # Provider prefix is stripped before reaching the bridge.
    assert call["model"] == "mistral-ocr-latest"
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    # Raw OCR params ride along in optional_params; Rust filters to supported keys.
    assert call["optional_params"].get("include_image_base64") is True


def test_ocr_skips_rust_when_disabled(monkeypatch, fake_bridge):
    """With the flag off, ocr() must take the normal Python provider path."""
    called = {}

    def _fake_handler(*_args, **_kwargs):
        called["hit"] = True
        return OCRResponse(pages=[], model="mistral-ocr-latest")

    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", _fake_handler)

    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert called.get("hit") is True
    assert fake_bridge == []  # Rust bridge never invoked


def test_ocr_resolves_key_via_secret_manager(monkeypatch, fake_bridge):
    """No explicit api_key: the Rust path must resolve it via get_secret_str so
    secret-manager backends (AWS/Azure/GCP/Vault) work, matching the Python path.
    Rust's own fallback only reads the process env, so Python resolves and passes it.
    """
    import litellm.secret_managers.main as secret_mgr

    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(secret_mgr, "get_secret_str", lambda name: "sk-from-vault")
    litellm.use_litellm_rust()

    litellm.ocr(model=MODEL, document=DOCUMENT)  # no api_key passed

    assert fake_bridge[0]["api_key"] == "sk-from-vault"


def test_ocr_forwards_timeout_to_rust(fake_bridge):
    """Caller-supplied timeout must flow into the Rust bridge so the fixed 600s
    client ceiling doesn't silently override shorter deadlines."""
    litellm.use_litellm_rust()

    litellm.ocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
        timeout=12.5,
    )

    assert fake_bridge[0]["timeout_seconds"] == 12.5


def test_ocr_passes_default_request_timeout_to_rust(fake_bridge):
    """When no explicit timeout is given, the library default (request_timeout)
    must still be forwarded so the Rust path matches the Python path's deadline."""
    from litellm.constants import request_timeout

    litellm.use_litellm_rust()

    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert fake_bridge[0]["timeout_seconds"] == float(request_timeout)


def test_ocr_runs_logging_on_rust_path(monkeypatch, fake_bridge):
    """The Rust shortcut must run the same logging setup (update_from_kwargs +
    pre_call) the Python path runs, otherwise callbacks and spend tracking break."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    update_calls = []
    pre_calls = []

    real_update = LiteLLMLoggingObj.update_from_kwargs
    real_pre = LiteLLMLoggingObj.pre_call

    def _record_update(self, *args, **kwargs):
        update_calls.append(kwargs)
        return real_update(self, *args, **kwargs)

    def _record_pre(self, *args, **kwargs):
        pre_calls.append(kwargs)
        return real_pre(self, *args, **kwargs)

    monkeypatch.setattr(LiteLLMLoggingObj, "update_from_kwargs", _record_update)
    monkeypatch.setattr(LiteLLMLoggingObj, "pre_call", _record_pre)
    litellm.use_litellm_rust()

    litellm.ocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
        include_image_base64=True,
    )

    assert update_calls, "update_from_kwargs must be invoked on the Rust path"
    assert update_calls[0].get("custom_llm_provider") == "mistral"
    assert update_calls[0].get("model") == "mistral-ocr-latest"
    assert pre_calls, "pre_call must be invoked on the Rust path"
    assert pre_calls[0].get("input") == "OCR document processing"
    assert pre_calls[0]["additional_args"]["complete_input_dict"]["document"] == DOCUMENT


def test_ocr_falls_back_to_python_when_bridge_missing(monkeypatch):
    """A missing ``litellm_python_bridge`` extension must degrade gracefully to
    the Python provider path instead of bubbling up ImportError."""
    monkeypatch.delitem(sys.modules, "litellm_python_bridge", raising=False)

    real_import_module = importlib.import_module

    def _blocked_import_module(name, package=None):
        if name == "litellm_python_bridge":
            raise ImportError("litellm_python_bridge not built")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _blocked_import_module)

    handler_calls = []

    def _fake_handler(*_args, **kwargs):
        handler_calls.append(kwargs)
        return OCRResponse(pages=[], model="mistral-ocr-latest")

    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", _fake_handler)
    litellm.use_litellm_rust()

    response = litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert isinstance(response, OCRResponse)
    assert handler_calls, "Python handler must run when the Rust bridge is missing"
