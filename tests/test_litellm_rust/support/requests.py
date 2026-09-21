from typing import Final

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm_rust.support.recording_server import RecordingServer

OCR_DOCUMENT: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}
OCR_MODEL: Final = "mistral/mistral-ocr-latest"
OCR_RESPONSE: Final = {
    "pages": [{"index": 0, "markdown": "native OCR response", "images": [], "dimensions": None}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 1, "doc_size_bytes": 3},
}

MESSAGES_MODEL: Final = "anthropic/claude-sonnet-5"
MESSAGES: Final = ({"role": "user", "content": "Hello"},)
MESSAGES_RESPONSE: Final = {
    "id": "msg_native",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "Hello from native Messages"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 4},
}
MESSAGES_EVENTS: Final = (
    ("message_start", {"type": "message_start", "message": {**MESSAGES_RESPONSE, "content": [], "stop_reason": None}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello from native Messages"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 4},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
)


def ocr_arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": OCR_MODEL,
        "document": dict(OCR_DOCUMENT),
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


def call_ocr(server: RecordingServer, **kwargs: object) -> OCRResponse:
    response: Final = litellm.ocr(**ocr_arguments(server, **kwargs))
    if not isinstance(response, OCRResponse):
        raise TypeError(f"Expected OCRResponse, got {type(response).__name__}")
    return response


async def call_aocr(server: RecordingServer, **kwargs: object) -> OCRResponse:
    return await litellm.aocr(**ocr_arguments(server, **kwargs))


def call_native_ocr(server: RecordingServer, **kwargs: object) -> OCRResponse:
    return call_ocr(server, **kwargs)


async def call_native_aocr(server: RecordingServer, **kwargs: object) -> OCRResponse:
    return await call_aocr(server, **kwargs)


async def call_native(server: RecordingServer, asynchronous: bool, **kwargs: object) -> OCRResponse:
    return await call_native_aocr(server, **kwargs) if asynchronous else call_native_ocr(server, **kwargs)


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
