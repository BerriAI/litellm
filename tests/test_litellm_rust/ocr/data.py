import json
from typing import Final

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests._support.provider import SharedProvider
from tests._support.wire import Reply

OCR_DOCUMENT: Final = {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"}
OCR_MODEL: Final = "mistral/mistral-ocr-latest"
OCR_TEXT: Final = "native OCR response"
OCR_RESPONSE: Final = {
    "pages": [{"index": 0, "markdown": OCR_TEXT, "images": [], "dimensions": None}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 1, "doc_size_bytes": 3},
}


def ocr_arguments(server: SharedProvider, **kwargs: object) -> dict[str, object]:
    return {
        "model": OCR_MODEL,
        "document": dict(OCR_DOCUMENT),
        "api_key": "test-key",
        "api_base": server.wire.url,
        **kwargs,
    }


def call_ocr(server: SharedProvider, **kwargs: object) -> OCRResponse:
    response: Final = litellm.ocr(**ocr_arguments(server, **kwargs))
    if not isinstance(response, OCRResponse):
        raise TypeError(f"Expected OCRResponse, got {type(response).__name__}")
    return response


async def call_aocr(server: SharedProvider, **kwargs: object) -> OCRResponse:
    return await litellm.aocr(**ocr_arguments(server, **kwargs))


async def call(server: SharedProvider, asynchronous: bool, **kwargs: object) -> OCRResponse:
    return await call_aocr(server, **kwargs) if asynchronous else call_ocr(server, **kwargs)


def ocr_reply(
    payload: object = OCR_RESPONSE,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> Reply:
    return Reply(body=json.dumps(payload).encode(), status=status, headers=headers or {})
