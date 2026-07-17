"""
Gateway E2E smoke for Rust-backed OCR.

Start the proxy with:

litellm --config tests/e2e/gateway/litellm-config.yml --port 4000
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
import yaml

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

RUST_OCR_GATEWAY_CASES = [
    pytest.param(
        "rust-ocr-mistral",
        {"type": "document_url", "document_url": TEST_PDF_URL},
        id="mistral",
    ),
    pytest.param(
        "rust-ocr-azure-ai",
        {"type": "document_url", "document_url": TEST_PDF_URL},
        id="azure_ai",
    ),
    pytest.param(
        "rust-ocr-azure-document-intelligence",
        {"type": "document_url", "document_url": TEST_PDF_URL},
        id="azure_document_intelligence",
    ),
    pytest.param(
        "rust-ocr-vertex-mistral",
        {"type": "document_url", "document_url": TEST_PDF_URL},
        id="vertex_mistral",
    ),
    pytest.param(
        "rust-ocr-vertex-deepseek",
        {
            "type": "image_url",
            "image_url": os.getenv("RUST_OCR_IMAGE_URL", TEST_IMAGE_URL),
        },
        id="vertex_deepseek",
    ),
]

SSRF_BLOCKED_DOCUMENT_URLS = [
    pytest.param("http://127.0.0.1/secret", id="loopback"),
    pytest.param("http://10.0.0.5/internal", id="rfc1918"),
    pytest.param("http://169.254.169.254/latest/meta-data/", id="link_local_metadata"),
    pytest.param("http://100.64.0.1/internal", id="cgnat"),
    pytest.param("http://198.18.0.1/internal", id="benchmark"),
    pytest.param("http://[::1]/secret", id="ipv6_loopback"),
    pytest.param("http://[::ffff:169.254.169.254]/x", id="mapped_ipv6_metadata"),
    pytest.param("http://[::ffff:10.0.0.5]/x", id="mapped_ipv6_rfc1918"),
]

CONFIG_PATH = Path(__file__).with_name("litellm-config.yml")


@dataclass(frozen=True)
class OcrGateway:
    base_url: str
    master_key: str

    def model_names(self) -> set[str]:
        with httpx.Client(
            timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120"))
        ) as client:
            response = client.get(
                f"{self.base_url.rstrip('/')}/model/info",
                headers={"Authorization": f"Bearer {self.master_key}"},
            )
        assert response.status_code == 200, response.text
        return {
            model["model_name"]
            for model in response.json().get("data", [])
            if "model_name" in model
        }

    def ocr(self, model: str, document: dict[str, str]) -> httpx.Response:
        with httpx.Client(
            timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120"))
        ) as client:
            return client.post(
                f"{self.base_url.rstrip('/')}/v1/ocr",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={"model": model, "document": document},
            )


@dataclass(frozen=True)
class OcrResources:
    gateway: OcrGateway


@pytest.fixture
def resources() -> OcrResources:
    proxy_url = os.getenv("LITELLM_PROXY_URL")
    if not proxy_url:
        pytest.skip(
            "Start a Rust OCR proxy and set LITELLM_PROXY_URL, e.g. http://localhost:4000"
        )
    return OcrResources(
        gateway=OcrGateway(
            base_url=proxy_url,
            master_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        )
    )


def _assert_ocr_response_shape(response_json: dict[str, Any]) -> None:
    assert response_json["object"] == "ocr"
    assert response_json["model"]
    assert isinstance(response_json["pages"], list)
    assert len(response_json["pages"]) > 0
    assert "index" in response_json["pages"][0]
    assert "markdown" in response_json["pages"][0]


class TestRustOcrGateway:
    def test_rust_ocr_models_are_on_gateway_config(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text())
        configured_models = {
            model_config["model_name"] for model_config in config["model_list"]
        }

        expected_models = {case.values[0] for case in RUST_OCR_GATEWAY_CASES}
        assert expected_models.issubset(configured_models)

    def test_running_gateway_loaded_rust_ocr_models(
        self, resources: OcrResources
    ) -> None:
        expected_models = {case.values[0] for case in RUST_OCR_GATEWAY_CASES}
        assert expected_models.issubset(resources.gateway.model_names())

    @pytest.mark.parametrize(("model", "document"), RUST_OCR_GATEWAY_CASES)
    def test_rust_ocr_model_gateway_response(
        self, resources: OcrResources, model: str, document: dict[str, str]
    ) -> None:
        response = resources.gateway.ocr(model, document)

        assert response.status_code == 200, response.text
        _assert_ocr_response_shape(response.json())

    @pytest.mark.parametrize("document_url", SSRF_BLOCKED_DOCUMENT_URLS)
    def test_rust_ocr_rejects_special_use_urls_with_typed_4xx(
        self, resources: OcrResources, document_url: str
    ) -> None:
        response = resources.gateway.ocr(
            "rust-ocr-azure-ai",
            {"type": "document_url", "document_url": document_url},
        )

        assert response.status_code == 400, response.text
        assert "SSRF protection" in response.text
        parsed = urlparse(document_url)
        assert document_url not in response.text
        assert parsed.netloc not in response.text
        if parsed.query:
            assert parsed.query not in response.text
