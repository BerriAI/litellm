import time
from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

DOCUMENT: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}
MODEL: Final = "mistral/mistral-ocr-latest"
OCR_RESPONSE: Final = {
    "pages": [{"index": 0, "markdown": "native OCR response", "images": [], "dimensions": None}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 1, "doc_size_bytes": 3},
}


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


def call_ocr(server: RecordingServer, **kwargs: object) -> OCRResponse:
    return litellm.ocr(
        model=MODEL,
        document=dict(DOCUMENT),
        api_key="test-key",
        api_base=server.base_url,
        **kwargs,
    )


def assert_native_request(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert not server.requests[0].headers.get("user-agent", "").startswith("python-httpx")


def test_ocr_sends_expected_provider_request(ocr_server: RecordingServer) -> None:
    response: Final = call_ocr(ocr_server)

    assert response.pages[0].markdown == "native OCR response"
    assert_native_request(ocr_server)
    assert ocr_server.requests[0].path == "/v1/ocr"
    assert ocr_server.requests[0].body == {"model": "mistral-ocr-latest", "document": DOCUMENT}


def test_ocr_rejects_unsupported_file_document_before_callbacks(ocr_server: RecordingServer) -> None:
    with pytest.raises(NotImplementedError, match="OCR file document preparation"):
        litellm.ocr(
            model=MODEL,
            document={"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"},
            api_key="test-key",
            api_base=ocr_server.base_url,
        )

    assert ocr_server.requests == []


def test_ocr_sends_optional_parameters(ocr_server: RecordingServer) -> None:
    call_ocr(ocr_server, pages=[0, 2], include_image_base64=True)

    assert ocr_server.requests[0].body["pages"] == [0, 2]
    assert ocr_server.requests[0].body["include_image_base64"] is True


def test_ocr_sends_custom_headers(ocr_server: RecordingServer) -> None:
    call_ocr(ocr_server, extra_headers={"x-trace-id": "trace-1"})

    assert ocr_server.requests[0].headers["authorization"] == "Bearer test-key"
    assert ocr_server.requests[0].headers["x-trace-id"] == "trace-1"


def test_ocr_resolves_provider_credentials(ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")

    litellm.ocr(model=MODEL, document=DOCUMENT, api_base=ocr_server.base_url)

    assert ocr_server.requests[0].headers["authorization"] == "Bearer environment-key"


def test_ocr_explicit_credentials_override_defaults(
    ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")

    call_ocr(ocr_server)

    assert ocr_server.requests[0].headers["authorization"] == "Bearer test-key"


def test_ocr_resolves_provider_endpoint(ocr_server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_AI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_AI_API_BASE", ocr_server.base_url)

    litellm.ocr(model="azure_ai/pixtral-12b-2409", document=DOCUMENT)

    assert_native_request(ocr_server)
    assert ocr_server.requests[0].path == "/providers/mistral/azure/ocr"
    assert ocr_server.requests[0].headers["api-key"] == "azure-key"


def test_ocr_resolves_vertex_project_and_location(ocr_server: RecordingServer) -> None:
    litellm.ocr(
        model="vertex_ai/mistral-ocr-2505",
        document=DOCUMENT,
        api_key="vertex-token",
        api_base=ocr_server.base_url,
        vertex_project="project-1",
        vertex_location="us-central1",
    )

    assert_native_request(ocr_server)
    assert ocr_server.requests[0].path == (
        "/v1/projects/project-1/locations/us-central1/publishers/mistralai/models/mistral-ocr-2505:rawPredict"
    )


def test_ocr_returns_normalized_response(ocr_server: RecordingServer) -> None:
    response: Final = call_ocr(ocr_server)

    assert isinstance(response, OCRResponse)
    assert response.model == "mistral-ocr-latest"
    assert response.usage_info.pages_processed == 1


def test_ocr_provider_error_preserves_status_and_context(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body={"message": "invalid OCR request"}, status=400))

    with pytest.raises(litellm.BadRequestError) as caught:
        call_ocr(ocr_server)

    assert caught.value.status_code == 400
    assert caught.value.model == "mistral-ocr-latest"
    assert caught.value.llm_provider == "mistral"
    assert "invalid OCR request" not in str(caught.value)


def test_ocr_honors_request_timeout(ocr_server: RecordingServer) -> None:
    ocr_server.enqueue(ResponseSpec(body=OCR_RESPONSE, delay=0.2))
    started_at: Final = time.monotonic()

    with pytest.raises(RuntimeError, match="OCR transport failed"):
        call_ocr(ocr_server, timeout=0.01)

    assert time.monotonic() - started_at < 0.15
    assert len(ocr_server.requests) == 1


def test_ocr_respects_runtime_toggle(ocr_server: RecordingServer) -> None:
    litellm.rust(False)
    call_ocr(ocr_server)
    litellm.rust(True)
    call_ocr(ocr_server)

    assert len(ocr_server.requests) == 2
    assert ocr_server.requests[0].headers["user-agent"].startswith("litellm/")
    assert ocr_server.requests[0].headers["accept-encoding"] != "identity"
    assert ocr_server.requests[1].headers["accept-encoding"] == "identity"
