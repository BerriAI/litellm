"""Live e2e: Rust-backed OCR is reachable through the gateway across providers.

The gateway config declares one rust-ocr deployment per provider (mistral,
azure_ai, azure document intelligence, vertex mistral, vertex deepseek). Start the
proxy with the Rust OCR path enabled:

    LITELLM_USE_RUST_OCR=1 litellm --config tests/e2e/gateway/litellm-config.yml

Three behaviors are checked: the config declares every provider's deployment (a
pure config read, no proxy needed); the running proxy loaded them onto /model/info;
and each one returns a well-formed OCR document over /v1/ocr. Per the e2e
"skip on environment, fail on behavior" rule, the proxy-backed cases skip when no
proxy answers but fail (never skip) once a request reaches it, so a provider whose
credentials are missing surfaces as a hard failure rather than silent green.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from e2e_http import unwrap
from models import OcrBody, OcrDocument, OcrResponse
from passthrough_client import PassthroughClient

# Tiny in-repo fixtures served via jsdelivr (sha-pinned, immutable) so the request
# bodies stay stable across runs.
TEST_PDF_URL = (
    "https://cdn.jsdelivr.net/gh/BerriAI/litellm"
    "@d769e81c90d453240c61fc572cdb27fae06a89d0"
    "/tests/llm_translation/fixtures/dummy.pdf"
)
TEST_IMAGE_URL = (
    "https://cdn.jsdelivr.net/gh/BerriAI/litellm"
    "@d769e81c90d453240c61fc572cdb27fae06a89d0"
    "/tests/image_gen_tests/test_image.png"
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "gateway" / "litellm-config.yml"


@dataclass(frozen=True, slots=True)
class _OcrCase:
    model: str
    document: OcrDocument


RUST_OCR_CASES: tuple[_OcrCase, ...] = (
    _OcrCase(
        "rust-ocr-mistral",
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
    ),
    _OcrCase(
        "rust-ocr-azure-ai",
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
    ),
    _OcrCase(
        "rust-ocr-azure-document-intelligence",
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
    ),
    _OcrCase(
        "rust-ocr-vertex-mistral",
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
    ),
    _OcrCase(
        "rust-ocr-vertex-deepseek",
        OcrDocument(type="image_url", image_url=TEST_IMAGE_URL),
    ),
)

_EXPECTED_MODELS = frozenset(case.model for case in RUST_OCR_CASES)
_CASE_IDS = tuple(case.model.removeprefix("rust-ocr-") for case in RUST_OCR_CASES)


class _ConfiguredModel(BaseModel):
    model_name: str


class _GatewayConfig(BaseModel):
    model_list: list[_ConfiguredModel]


def _configured_model_names() -> frozenset[str]:
    config = _GatewayConfig.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
    return frozenset(entry.model_name for entry in config.model_list)


def _assert_ocr_document(response: OcrResponse) -> None:
    assert response.object == "ocr", f"expected object='ocr', got {response.object!r}"
    assert response.model, "response missing the resolved model name"
    assert response.pages, "OCR returned no pages"
    assert response.pages[0].markdown is not None, "first page has no markdown"


def test_rust_ocr_models_declared_in_gateway_config() -> None:
    """Pure config read (no proxy): every provider's rust-ocr deployment the suite
    exercises is declared in the gateway config the proxy runs with. A case added
    here without a matching deployment fails before any live call is attempted."""
    missing = _EXPECTED_MODELS - _configured_model_names()
    assert not missing, f"rust-ocr models absent from {CONFIG_PATH.name}: {missing}"


@pytest.mark.e2e
class TestRustOcrGateway:
    def test_gateway_loaded_rust_ocr_models(self, client: PassthroughClient) -> None:
        loaded = frozenset(entry.model_name for entry in client.gateway.model_info())
        missing = _EXPECTED_MODELS - loaded
        assert not missing, f"proxy did not load rust-ocr models: {missing}"

    @pytest.mark.parametrize("case", RUST_OCR_CASES, ids=_CASE_IDS)
    def test_rust_ocr_response(self, client: PassthroughClient, scoped_key: str, case: _OcrCase) -> None:
        response = unwrap(client.gateway.ocr(scoped_key, OcrBody(model=case.model, document=case.document)))
        _assert_ocr_document(response)
