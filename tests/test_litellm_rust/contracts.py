from typing import Final

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import ocr as native_ocr
from tests.test_litellm_rust.recording_server import RecordingServer

OCR_DOCUMENT: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}
OCR_MODEL: Final = "mistral/mistral-ocr-latest"
OCR_RESPONSE: Final = {
    "pages": [{"index": 0, "markdown": "native OCR response", "images": [], "dimensions": None}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 1, "doc_size_bytes": 3},
}

MESSAGES_MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"
MESSAGES: Final = [{"role": "user", "content": "Hello"}]
MESSAGES_RESPONSE: Final = {
    "id": "msg_native",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "Hello from native Messages"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 4},
}


def ocr_arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": OCR_MODEL,
        "document": dict(OCR_DOCUMENT),
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


def call_native_ocr(server: RecordingServer, **kwargs: object) -> OCRResponse:
    return native_ocr.ocr(ocr_arguments(server, **kwargs))


async def call_native_aocr(server: RecordingServer, **kwargs: object) -> OCRResponse:
    return await native_ocr.aocr(ocr_arguments(server, **kwargs))


def request_body(kwargs: dict[str, object]) -> dict[str, object]:
    additional_args = kwargs["additional_args"]
    assert isinstance(additional_args, dict)
    body = additional_args["complete_input_dict"]
    assert isinstance(body, dict)
    return body


def request_headers(kwargs: dict[str, object]) -> dict[str, object]:
    additional_args = kwargs["additional_args"]
    assert isinstance(additional_args, dict)
    headers = additional_args["headers"]
    assert isinstance(headers, dict)
    return headers
