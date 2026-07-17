"""
Gateway E2E smoke for Rust-backed OCR.

Start the proxy with:

litellm --config tests/e2e/gateway/litellm-config.yml --port 4000
"""

from __future__ import annotations

import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator, Literal, TextIO, cast

import httpx
import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field

from litellm.rust_bridge import native_bridge_available

REPO_ROOT = Path(__file__).resolve().parents[3]

_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]+|Bearer\s+\S+)")

_CAPTURE_RESPONSE_BODY: bytes = json.dumps(
    {
        "pages": [{"index": 0, "markdown": "captured"}],
        "model": "mistral-ocr-latest",
        "usage_info": {"pages_processed": 1},
    }
).encode()


class OcrDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    document_url: str | None = None
    image_url: str | None = None


class JsonSchemaProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str


class AnnotationJsonSchemaBody(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    type: str
    properties: dict[str, JsonSchemaProperty]
    required: tuple[str, ...]
    additional_properties: bool = Field(alias="additionalProperties")


class AnnotationJsonSchema(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str
    body: AnnotationJsonSchemaBody = Field(alias="schema")
    strict: bool


class MistralAnnotationFormat(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    json_schema: AnnotationJsonSchema


class MistralOcrParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    pages: tuple[int, ...]
    include_image_base64: bool
    include_blocks: bool
    image_limit: int
    image_min_size: int
    bbox_annotation_format: MistralAnnotationFormat
    document_annotation_format: MistralAnnotationFormat
    document_annotation_prompt: str
    extract_header: bool
    extract_footer: bool
    table_format: str
    confidence_scores_granularity: str
    id: str


class TraceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace: str


class LitellmInternalCanaries(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: TraceMetadata
    litellm_metadata: TraceMetadata
    num_retries: int
    tags: tuple[str, ...]
    litellm_session_id: str
    original_generic_function: str


class MistralOcrUpstreamRequest(MistralOcrParams):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    document: OcrDocument


class OcrResponsePage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    index: int
    markdown: str


class OcrUsageInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    pages_processed: int


class OcrResponseEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    object: Literal["ocr"]
    model: str = Field(min_length=1)
    pages: tuple[OcrResponsePage, ...] = Field(min_length=1)
    usage_info: OcrUsageInfo | None = None


class ModelInfoEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    model_name: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    data: tuple[ModelInfoEntry, ...]


class GatewayConfigEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    model_name: str


class GatewayConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    model_list: tuple[GatewayConfigEntry, ...]


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

CAPTURE_DOCUMENT = OcrDocument(type="document_url", document_url=TEST_PDF_URL)

SUPPORTED_PARAMS = MistralOcrParams(
    pages=(0,),
    include_image_base64=True,
    include_blocks=True,
    image_limit=10,
    image_min_size=64,
    bbox_annotation_format=MistralAnnotationFormat(
        type="json_schema",
        json_schema=AnnotationJsonSchema(
            name="bbox_annotation",
            schema=AnnotationJsonSchemaBody(
                type="object",
                properties={"description": JsonSchemaProperty(type="string")},
                required=("description",),
                additionalProperties=False,
            ),
            strict=True,
        ),
    ),
    document_annotation_format=MistralAnnotationFormat(
        type="json_schema",
        json_schema=AnnotationJsonSchema(
            name="document_annotation",
            schema=AnnotationJsonSchemaBody(
                type="object",
                properties={"title": JsonSchemaProperty(type="string")},
                required=("title",),
                additionalProperties=False,
            ),
            strict=True,
        ),
    ),
    document_annotation_prompt="extract the title",
    extract_header=True,
    extract_footer=False,
    table_format="markdown",
    confidence_scores_granularity="word",
    id="ocr-req-parity-9",
)

INTERNAL_CANARIES = LitellmInternalCanaries(
    metadata=TraceMetadata(trace="internal"),
    litellm_metadata=TraceMetadata(trace="internal"),
    num_retries=3,
    tags=("internal",),
    litellm_session_id="sess-internal",
    original_generic_function="litellm-internal-should-be-filtered",
)

EXPECTED_UPSTREAM = MistralOcrUpstreamRequest(
    model="mistral-ocr-latest",
    document=CAPTURE_DOCUMENT,
    pages=SUPPORTED_PARAMS.pages,
    include_image_base64=SUPPORTED_PARAMS.include_image_base64,
    include_blocks=SUPPORTED_PARAMS.include_blocks,
    image_limit=SUPPORTED_PARAMS.image_limit,
    image_min_size=SUPPORTED_PARAMS.image_min_size,
    bbox_annotation_format=SUPPORTED_PARAMS.bbox_annotation_format,
    document_annotation_format=SUPPORTED_PARAMS.document_annotation_format,
    document_annotation_prompt=SUPPORTED_PARAMS.document_annotation_prompt,
    extract_header=SUPPORTED_PARAMS.extract_header,
    extract_footer=SUPPORTED_PARAMS.extract_footer,
    table_format=SUPPORTED_PARAMS.table_format,
    confidence_scores_granularity=SUPPORTED_PARAMS.confidence_scores_granularity,
    id=SUPPORTED_PARAMS.id,
)

RUST_OCR_GATEWAY_CASES = [
    pytest.param(
        "rust-ocr-mistral",
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
        id="mistral",
    ),
    pytest.param(
        "rust-ocr-azure-ai",
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
        id="azure_ai",
    ),
    pytest.param(
        "rust-ocr-azure-document-intelligence",
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
        id="azure_document_intelligence",
    ),
    pytest.param(
        "rust-ocr-vertex-mistral",
        OcrDocument(type="document_url", document_url=TEST_PDF_URL),
        id="vertex_mistral",
    ),
    pytest.param(
        "rust-ocr-vertex-deepseek",
        OcrDocument(
            type="image_url",
            image_url=os.getenv("RUST_OCR_IMAGE_URL", TEST_IMAGE_URL),
        ),
        id="vertex_deepseek",
    ),
]

CONFIG_PATH = Path(__file__).with_name("litellm-config.yml")


def _wire_payload(
    model: str,
    document: OcrDocument,
    params: MistralOcrParams | None,
    canaries: LitellmInternalCanaries | None = None,
) -> str:
    return json.dumps(
        {
            "model": model,
            "document": document.model_dump(mode="json", exclude_none=True),
            **(
                params.model_dump(mode="json", by_alias=True)
                if params is not None
                else {}
            ),
            **(
                canaries.model_dump(mode="json", by_alias=True)
                if canaries is not None
                else {}
            ),
        }
    )


@dataclass(frozen=True)
class OcrGateway:
    base_url: str
    master_key: str

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120")))

    def model_names(self) -> frozenset[str]:
        with self._client() as client:
            response = client.get(
                f"{self.base_url.rstrip('/')}/model/info",
                headers={"Authorization": f"Bearer {self.master_key}"},
            )
        assert response.status_code == 200, response.text
        parsed = ModelInfoResponse.model_validate_json(response.content)
        return frozenset(entry.model_name for entry in parsed.data)

    def ocr(
        self,
        model: str,
        document: OcrDocument,
        params: MistralOcrParams | None = None,
    ) -> httpx.Response:
        with self._client() as client:
            return client.post(
                f"{self.base_url.rstrip('/')}/v1/ocr",
                headers={
                    "Authorization": f"Bearer {self.master_key}",
                    "content-type": "application/json",
                },
                content=_wire_payload(model, document, params),
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


class TestRustOcrGateway:
    def test_rust_ocr_models_are_on_gateway_config(self) -> None:
        config = GatewayConfig.model_validate(
            cast(object, yaml.safe_load(CONFIG_PATH.read_text()))
        )
        configured_models = frozenset(entry.model_name for entry in config.model_list)

        expected_models = {str(case.values[0]) for case in RUST_OCR_GATEWAY_CASES}
        assert expected_models.issubset(configured_models)

    def test_running_gateway_loaded_rust_ocr_models(
        self, resources: OcrResources
    ) -> None:
        expected_models = {str(case.values[0]) for case in RUST_OCR_GATEWAY_CASES}
        assert expected_models.issubset(resources.gateway.model_names())

    @pytest.mark.parametrize(("model", "document"), RUST_OCR_GATEWAY_CASES)
    def test_rust_ocr_model_gateway_response(
        self, resources: OcrResources, model: str, document: OcrDocument
    ) -> None:
        response = resources.gateway.ocr(model, document)

        assert response.status_code == 200, response.text
        OcrResponseEnvelope.model_validate_json(response.content)

    @pytest.mark.e2e
    def test_rust_ocr_mistral_live_forwards_supported_params(
        self, resources: OcrResources
    ) -> None:
        if not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY not set for live Mistral OCR call")

        response = resources.gateway.ocr(
            "rust-ocr-mistral", CAPTURE_DOCUMENT, SUPPORTED_PARAMS
        )

        assert response.status_code == 200, response.text
        parsed = OcrResponseEnvelope.model_validate_json(response.content)
        assert parsed.pages[0].markdown != ""
        if parsed.usage_info is not None:
            assert parsed.usage_info.pages_processed >= 1


def _make_capture_handler(
    captures: queue.Queue[bytes],
) -> type[BaseHTTPRequestHandler]:
    class _CaptureHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            captures.put(self.rfile.read(length))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(_CAPTURE_RESPONSE_BODY)))
            self.end_headers()
            self.wfile.write(_CAPTURE_RESPONSE_BODY)

        def log_message(self, format: str, *args: object) -> None:
            return

    return _CaptureHandler


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = cast(tuple[str, int], sock.getsockname())
        return port


def _wait_for_liveness(base_url: str, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health/liveliness", timeout=2)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


@dataclass(frozen=True)
class _CaptureProxy:
    proxy_url: str
    captures: queue.Queue[bytes]


def _sanitize(text: str) -> str:
    return _SECRET_PATTERN.sub("[redacted]", text)


@pytest.fixture
def capture_proxy(tmp_path: Path) -> Iterator[_CaptureProxy]:
    if not native_bridge_available():
        pytest.skip("compiled Rust OCR bridge is required for the capture E2E")

    captures: queue.Queue[bytes] = queue.Queue()
    capture_port = _free_port()
    capture_server = HTTPServer(
        ("127.0.0.1", capture_port), _make_capture_handler(captures)
    )
    server_thread = threading.Thread(target=capture_server.serve_forever, daemon=True)
    server_thread.start()

    proxy: subprocess.Popen[bytes] | None = None
    proxy_log: TextIO | None = None
    proxy_log_path = tmp_path / "capture-proxy.log"
    try:
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

        proxy_log = proxy_log_path.open("w")
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
            stdout=proxy_log,
            stderr=subprocess.STDOUT,
        )
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        if not _wait_for_liveness(proxy_url, time.monotonic() + 90):
            proxy_log.flush()
            tail = _sanitize(proxy_log_path.read_text()[-4000:])
            pytest.fail(
                f"capture proxy did not become live while the Rust bridge is available; "
                f"sanitized proxy log at {proxy_log_path}\n{tail}"
            )
        yield _CaptureProxy(proxy_url=proxy_url, captures=captures)
    finally:
        if proxy is not None:
            proxy.terminate()
            try:
                proxy.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait()
        if proxy_log is not None:
            proxy_log.close()
        capture_server.shutdown()
        capture_server.server_close()
        server_thread.join(timeout=5)


def test_rust_ocr_proxy_forwards_full_contract_to_capture_endpoint(
    capture_proxy: _CaptureProxy,
) -> None:
    response = httpx.post(
        f"{capture_proxy.proxy_url}/v1/ocr",
        headers={
            "Authorization": "Bearer sk-1234",
            "content-type": "application/json",
        },
        content=_wire_payload(
            "rust-ocr-mistral-capture",
            CAPTURE_DOCUMENT,
            SUPPORTED_PARAMS,
            INTERNAL_CANARIES,
        ),
        timeout=60,
    )
    assert response.status_code == 200, response.text
    OcrResponseEnvelope.model_validate_json(response.content)

    captured = MistralOcrUpstreamRequest.model_validate_json(
        capture_proxy.captures.get(timeout=10)
    )
    assert captured == EXPECTED_UPSTREAM
