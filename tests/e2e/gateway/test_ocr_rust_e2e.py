"""
Gateway E2E smoke for Rust-backed OCR.

Start the proxy with:

litellm --config tests/e2e/gateway/litellm-config.yml --port 4000
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

MISTRAL_OCR_SUPPORTED_PARAMS: dict[str, Any] = {
    "pages": [0, 2, 5],
    "include_image_base64": True,
    "include_blocks": True,
    "image_limit": 10,
    "image_min_size": 64,
    "bbox_annotation_format": {"type": "text"},
    "document_annotation_format": {"type": "json_schema"},
    "document_annotation_prompt": "extract the title",
    "extract_header": True,
    "extract_footer": False,
    "table_format": "html",
    "confidence_scores_granularity": "word",
    "id": "ocr-req-parity-9",
}

LITELLM_INTERNAL_CANARIES: dict[str, Any] = {
    "metadata": {"trace": "internal"},
    "litellm_metadata": {"trace": "internal"},
    "num_retries": 3,
    "tags": ["internal"],
    "litellm_session_id": "sess-internal",
}

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

    def ocr(
        self,
        model: str,
        document: dict[str, str],
        extra_params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        payload: dict[str, Any] = {"model": model, "document": document}
        if extra_params:
            payload.update(extra_params)
        with httpx.Client(
            timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120"))
        ) as client:
            return client.post(
                f"{self.base_url.rstrip('/')}/v1/ocr",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json=payload,
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

    @pytest.mark.e2e
    def test_rust_ocr_mistral_live_forwards_supported_params(
        self, resources: OcrResources
    ) -> None:
        if not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY not set for live Mistral OCR call")

        provider_params = {
            key: value
            for key, value in MISTRAL_OCR_SUPPORTED_PARAMS.items()
            if key in {"include_image_base64", "pages", "id"}
        }
        response = resources.gateway.ocr(
            "rust-ocr-mistral",
            {"type": "document_url", "document_url": TEST_PDF_URL},
            extra_params=provider_params,
        )

        assert response.status_code == 200, response.text
        _assert_ocr_response_shape(response.json())


class _CaptureHandler(BaseHTTPRequestHandler):
    captured_body: dict[str, Any] | None = None

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        type(self).captured_body = json.loads(self.rfile.read(length))
        body = json.dumps(
            {
                "pages": [{"index": 0, "markdown": "captured"}],
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: Any) -> None:
        return


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_liveness(base_url: str, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health/liveliness", timeout=2)
            if resp.status_code < 500:
                return True
        except httpx.HTTPError:
            time.sleep(0.5)
    return False


@dataclass(frozen=True)
class _CaptureProxy:
    proxy_url: str
    capture: type[_CaptureHandler]


@pytest.fixture
def capture_proxy(tmp_path: Path) -> Iterator[_CaptureProxy]:
    native_available = __import__(
        "litellm.rust_bridge", fromlist=["native_bridge_available"]
    ).native_bridge_available()
    if not native_available:
        pytest.skip("compiled Rust OCR bridge is required for the capture E2E")

    capture_port = _free_port()
    capture_server = HTTPServer(("127.0.0.1", capture_port), _CaptureHandler)
    _CaptureHandler.captured_body = None
    threading.Thread(target=capture_server.serve_forever, daemon=True).start()

    proxy_port = _free_port()
    config = {
        "model_list": [
            {
                "model_name": "rust-ocr-mistral-capture",
                "litellm_params": {
                    "model": "mistral/mistral-ocr-latest",
                    "api_key": "sk-capture-test",
                    "api_base": f"http://127.0.0.1:{capture_port}",
                },
            }
        ],
        "general_settings": {"master_key": "sk-1234"},
        "litellm_settings": {"drop_params": False},
    }
    config_path = tmp_path / "capture-config.yml"
    config_path.write_text(yaml.safe_dump(config))

    proxy = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "litellm" / "proxy" / "proxy_cli.py"),
            "--config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(proxy_port),
            "--num_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    try:
        if not _wait_for_liveness(proxy_url, time.monotonic() + 90):
            pytest.skip("capture proxy did not become live in time")
        yield _CaptureProxy(proxy_url=proxy_url, capture=_CaptureHandler)
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proxy.kill()
        capture_server.shutdown()


def test_rust_ocr_proxy_forwards_full_contract_to_capture_endpoint(
    capture_proxy: _CaptureProxy,
) -> None:
    payload: dict[str, Any] = {
        "model": "rust-ocr-mistral-capture",
        "document": {"type": "document_url", "document_url": TEST_PDF_URL},
        **MISTRAL_OCR_SUPPORTED_PARAMS,
        **LITELLM_INTERNAL_CANARIES,
    }
    response = httpx.post(
        f"{capture_proxy.proxy_url}/v1/ocr",
        headers={"Authorization": "Bearer sk-1234"},
        json=payload,
        timeout=60,
    )
    assert response.status_code == 200, response.text
    _assert_ocr_response_shape(response.json())

    captured = capture_proxy.capture.captured_body
    assert captured is not None, "capture endpoint never received the upstream request"

    for key, value in MISTRAL_OCR_SUPPORTED_PARAMS.items():
        assert captured.get(key) == value, f"supported param {key} missing/wrong upstream"

    leaked = [key for key in LITELLM_INTERNAL_CANARIES if key in captured]
    assert not leaked, f"internal params leaked upstream: {leaked}"

    assert captured["model"] == "mistral-ocr-latest"
    assert captured["document"] == payload["document"]
