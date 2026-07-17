"""
Gateway E2E smoke for Rust-backed OCR.

Start the proxy with:

litellm --config tests/e2e/gateway/litellm-config.yml --port 4000
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import yaml

from litellm.llms.base_llm.ocr.transformation import OCRResponse

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

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PDF = REPO_ROOT / "tests" / "llm_translation" / "fixtures" / "dummy.pdf"
FIXTURE_IMAGE = REPO_ROOT / "tests" / "image_gen_tests" / "test_image.png"

RUST_OCR_UPLOAD_CASES = (
    pytest.param(FIXTURE_PDF, id="pdf_octet_stream"),
    pytest.param(FIXTURE_IMAGE, id="image_octet_stream"),
)

RUST_OCR_GATEWAY_CASES = (
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
)

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

    def ocr_upload(
        self, model: str, content: bytes, upload_name: str
    ) -> httpx.Response:
        with httpx.Client(
            timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120"))
        ) as client:
            return client.post(
                f"{self.base_url.rstrip('/')}/v1/ocr",
                headers={"Authorization": f"Bearer {self.master_key}"},
                data={"model": model},
                files={"file": (upload_name, content, "application/octet-stream")},
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


def _assert_ocr_response(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    parsed = OCRResponse.model_validate(response.json())
    assert parsed.object == "ocr"
    assert parsed.model
    assert len(parsed.pages) > 0
    assert any(page.markdown.strip() for page in parsed.pages)


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

        _assert_ocr_response(response)

    @pytest.mark.parametrize("fixture_path", RUST_OCR_UPLOAD_CASES)
    def test_rust_ocr_octet_stream_upload_response(
        self,
        resources: OcrResources,
        fixture_path: Path,
    ) -> None:
        assert fixture_path.is_file(), f"Missing committed OCR fixture: {fixture_path}"

        content = fixture_path.read_bytes()
        response = resources.gateway.ocr_upload("rust-ocr-mistral", content, "document")

        _assert_ocr_response(response)
