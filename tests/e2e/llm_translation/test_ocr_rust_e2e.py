"""Live e2e: Rust-backed OCR is reachable through the gateway across providers.

Each provider's OCR deployment is registered at runtime via /model/new and deleted
on teardown, so nothing is hardcoded into the gateway config. Every provider is its
own typed OcrProvider below: it owns the model id and the os.environ/* credential
references the proxy resolves at call time, so adding a provider is a new type
rather than another inline body. Start the proxy with the Rust OCR path enabled:

Each case creates its deployment, drives a real /v1/ocr call, and asserts a
well-formed OCR document comes back. Per the e2e "skip on environment, fail on
behavior" rule, a case skips when no proxy answers but fails (never skips) once a
request reaches it: the proxy fetches each provider's referenced secrets, so a
missing credential surfaces as a live provider error rather than silent green.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, OcrBody, OcrDocument, OcrResponse

pytestmark = pytest.mark.e2e

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


class OcrProvider(Protocol):
    """One OCR provider's deployment config: its model id plus the os.environ/*
    credential references the proxy resolves at call time. Each provider owns which
    env vars it reads, so a new provider is a new type, not another inline body."""

    def litellm_params(self) -> LiteLLMParamsBody: ...


@dataclass(frozen=True, slots=True)
class MistralOcr:
    model: str = "mistral/mistral-ocr-latest"

    def litellm_params(self) -> LiteLLMParamsBody:
        return LiteLLMParamsBody(model=self.model, api_key="os.environ/MISTRAL_API_KEY")


@dataclass(frozen=True, slots=True)
class AzureAiOcr:
    """azure_ai (mistral) OCR. The rust OCR path resolves credentials itself from
    AZURE_AI_API_BASE / AZURE_AI_API_KEY when the deployment leaves them unset; it
    does NOT unwrap an `os.environ/*` reference passed as api_base (it would be sent
    to Azure verbatim), so we omit them and let litellm read the env vars by name."""

    model: str

    def litellm_params(self) -> LiteLLMParamsBody:
        return LiteLLMParamsBody(model=self.model)


@dataclass(frozen=True, slots=True)
class AzureDocIntelligenceOcr:
    """azure_ai Document Intelligence OCR. A separate Azure resource from the
    mistral one, so it has its own env vars: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT /
    AZURE_DOCUMENT_INTELLIGENCE_API_KEY, which the OCR config resolves from the
    doc-intelligence model name when api_base/api_key are left unset."""

    model: str = "azure_ai/doc-intelligence/prebuilt-layout"

    def litellm_params(self) -> LiteLLMParamsBody:
        return LiteLLMParamsBody(model=self.model)


@dataclass(frozen=True, slots=True)
class VertexOcr:
    """Vertex AI OCR (Mistral publisher). Only the location (not a secret) is set;
    the project and credentials are left unset so the gateway resolves VERTEXAI_PROJECT
    and VERTEXAI_CREDENTIALS from its own environment by name, keeping every secret on
    the gateway like the azure_ai cases above. This is deliberate: the OCR path reads
    vertex_project verbatim from litellm_params and never unwraps an `os.environ/*`
    ref, so passing one would put the literal string in the request URL."""

    model: str
    location: str

    def litellm_params(self) -> LiteLLMParamsBody:
        return LiteLLMParamsBody(model=self.model, vertex_location=self.location)


@dataclass(frozen=True, slots=True)
class _OcrCase:
    suffix: str
    provider: OcrProvider
    document: OcrDocument


RUST_OCR_CASES: tuple[_OcrCase, ...] = (
    _OcrCase(
        "mistral",
        MistralOcr(),
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
    ),
    _OcrCase(
        "azure-ai",
        AzureAiOcr("azure_ai/mistral-document-ai-2512"),
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
    ),
    _OcrCase(
        "azure-document-intelligence",
        AzureDocIntelligenceOcr(),
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
    ),
    _OcrCase(
        "vertex-mistral",
        VertexOcr("vertex_ai/mistral-ocr-2505", "us-central1"),
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
    ),
)

_CASE_IDS = tuple(case.suffix for case in RUST_OCR_CASES)


def _assert_ocr_document(response: OcrResponse) -> None:
    assert response.object == "ocr", f"expected object='ocr', got {response.object!r}"
    assert response.model, "response missing the resolved model name"
    assert response.pages, "OCR returned no pages"
    assert response.pages[0].markdown is not None, "first page has no markdown"


class TestRustOcrGateway:
    @pytest.mark.parametrize("case", RUST_OCR_CASES, ids=_CASE_IDS)
    def test_rust_ocr_response(
        self, endpoints_client: EndpointsClient, resources: ResourceManager, case: _OcrCase
    ) -> None:
        model = f"rust-ocr-{case.suffix}-{unique_marker()}"
        model_id = endpoints_client.create_model(model, case.provider.litellm_params())
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        response = unwrap(endpoints_client.gateway.ocr(key, OcrBody(model=model, document=case.document)))
        _assert_ocr_document(response)


