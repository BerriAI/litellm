"""
Opt-in live E2E coverage for Rust-backed OCR providers.

These tests are intentionally reusable across OCR providers. They validate the
same Mistral-compatible response contract through:
1. the Python SDK with ``litellm.use_litellm_rust(True)``; and
2. a running LiteLLM proxy started with ``LITELLM_USE_RUST_OCR=1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

import litellm
from base_ocr_unit_tests import TEST_PDF_URL
from litellm.ocr.rust_bridge import load_rust_aocr

TEST_IMAGE_URL = (
    "https://cdn.jsdelivr.net/gh/BerriAI/litellm"
    "@d769e81c90d453240c61fc572cdb27fae06a89d0"
    "/tests/image_gen_tests/test_image.png"
)


@dataclass(frozen=True)
class RustOcrE2ECase:
    name: str
    model: str
    document: dict[str, str]
    required_env: tuple[str, ...]
    optional_env: tuple[str, ...] = ()

    def litellm_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model}
        for env_name in self.required_env + self.optional_env:
            value = os.getenv(env_name)
            if value is not None:
                kwargs.update(_env_to_litellm_kwargs(env_name, value))
        return kwargs


def _env_to_litellm_kwargs(env_name: str, value: str) -> dict[str, str]:
    if env_name in {
        "MISTRAL_API_KEY",
        "AZURE_API_KEY",
        "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
    }:
        return {"api_key": value}
    if env_name in {"AZURE_API_BASE", "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"}:
        return {"api_base": value}
    if env_name in {"VERTEXAI_PROJECT", "VERTEX_AI_PROJECT"}:
        return {"vertex_project": value}
    if env_name in {"VERTEXAI_LOCATION", "VERTEX_LOCATION", "VERTEX_AI_LOCATION"}:
        return {"vertex_location": value}
    return {}


RUST_OCR_E2E_CASES = [
    RustOcrE2ECase(
        name="mistral",
        model="mistral/mistral-ocr-latest",
        document={"type": "document_url", "document_url": TEST_PDF_URL},
        required_env=("MISTRAL_API_KEY",),
    ),
    RustOcrE2ECase(
        name="azure_ai",
        model="azure_ai/mistral-document-ai-2505",
        document={"type": "document_url", "document_url": TEST_PDF_URL},
        required_env=("AZURE_API_KEY", "AZURE_API_BASE"),
    ),
    RustOcrE2ECase(
        name="azure_document_intelligence",
        model="azure_ai/doc-intelligence/prebuilt-layout",
        document={"type": "document_url", "document_url": TEST_PDF_URL},
        required_env=(
            "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        ),
    ),
    RustOcrE2ECase(
        name="vertex_mistral",
        model="vertex_ai/mistral-ocr-2505",
        document={"type": "document_url", "document_url": TEST_PDF_URL},
        required_env=("VERTEXAI_PROJECT",),
        optional_env=("VERTEXAI_LOCATION", "VERTEX_LOCATION"),
    ),
    RustOcrE2ECase(
        name="vertex_deepseek",
        model="vertex_ai/deepseek-ocr-maas",
        document={
            "type": "image_url",
            "image_url": os.getenv("RUST_OCR_IMAGE_URL", TEST_IMAGE_URL),
        },
        required_env=("VERTEXAI_PROJECT",),
        optional_env=("VERTEXAI_LOCATION", "VERTEX_LOCATION"),
    ),
]


def _skip_unless_live_rust_ocr_e2e(case: RustOcrE2ECase) -> None:
    if os.getenv("LITELLM_RUN_LIVE_RUST_OCR_TESTS") != "1":
        pytest.skip(
            "Set LITELLM_RUN_LIVE_RUST_OCR_TESTS=1 to run live Rust OCR E2E tests"
        )
    missing = [env_name for env_name in case.required_env if not os.getenv(env_name)]
    if missing:
        pytest.skip(f"Missing required env vars for {case.name}: {', '.join(missing)}")
    if case.name.startswith("vertex_") and not (
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("VERTEXAI_CREDENTIALS")
        or os.getenv("VERTEX_AI_API_KEY")
        or os.getenv("VERTEXAI_API_KEY")
    ):
        pytest.skip(
            "Vertex OCR Rust E2E requires GOOGLE_APPLICATION_CREDENTIALS, VERTEXAI_CREDENTIALS, VERTEX_AI_API_KEY, or VERTEXAI_API_KEY"
        )


def _assert_ocr_response_shape(response: Any) -> None:
    assert response.object == "ocr"
    assert response.model
    assert isinstance(response.pages, list)
    assert len(response.pages) > 0
    assert hasattr(response.pages[0], "index")
    assert hasattr(response.pages[0], "markdown")


def _assert_proxy_ocr_response_shape(response_json: dict[str, Any]) -> None:
    assert response_json["object"] == "ocr"
    assert response_json["model"]
    assert isinstance(response_json["pages"], list)
    assert len(response_json["pages"]) > 0
    assert "index" in response_json["pages"][0]
    assert "markdown" in response_json["pages"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", RUST_OCR_E2E_CASES, ids=lambda case: case.name)
async def test_rust_ocr_provider_live_sdk(case: RustOcrE2ECase) -> None:
    _skip_unless_live_rust_ocr_e2e(case)
    litellm.use_litellm_rust(True)
    if load_rust_aocr() is None:
        pytest.skip("compiled litellm_python_bridge.aocr is required for Rust OCR E2E")

    try:
        response = await litellm.aocr(document=case.document, **case.litellm_kwargs())
        _assert_ocr_response_shape(response)
    finally:
        litellm.use_litellm_rust(False)


@pytest.mark.parametrize("case", RUST_OCR_E2E_CASES, ids=lambda case: case.name)
def test_rust_ocr_provider_live_proxy_gateway(case: RustOcrE2ECase) -> None:
    _skip_unless_live_rust_ocr_e2e(case)
    proxy_url = os.getenv("LITELLM_PROXY_URL")
    if not proxy_url:
        pytest.skip("Set LITELLM_PROXY_URL to run proxy OCR E2E")
    if os.getenv("LITELLM_USE_RUST_OCR") != "1":
        pytest.skip(
            "Start the proxy with LITELLM_USE_RUST_OCR=1 for Rust OCR proxy E2E"
        )

    master_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
    body = {
        **case.litellm_kwargs(),
        "document": case.document,
    }
    with httpx.Client(timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120"))) as client:
        try:
            health = client.get(f"{proxy_url.rstrip('/')}/health/liveliness")
        except httpx.HTTPError as exc:
            pytest.skip(f"No live proxy at {proxy_url}: {exc}")
        if health.status_code >= 500:
            pytest.skip(f"Proxy at {proxy_url} returned {health.status_code}")

        response = client.post(
            f"{proxy_url.rstrip('/')}/v1/ocr",
            headers={"Authorization": f"Bearer {master_key}"},
            json=body,
        )

    assert response.status_code == 200, response.text
    _assert_proxy_ocr_response_shape(response.json())
