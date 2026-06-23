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

    def _ocr(model, document, api_key, api_base, optional_params):
        calls.append(
            {
                "model": model,
                "document": document,
                "api_key": api_key,
                "api_base": api_base,
                "optional_params": optional_params,
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
