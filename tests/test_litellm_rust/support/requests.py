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
