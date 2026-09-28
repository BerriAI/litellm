from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.catalog import Route
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.support.callback_contract import CallbackRoute
from tests.test_litellm_rust.support.requests import (
    MESSAGES,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    OCR_DOCUMENT,
    OCR_MODEL,
    OCR_RESPONSE,
)

OBJECT: Final = TypeAdapter(dict[str, object])
OBJECTS: Final = TypeAdapter(list[dict[str, object]])


def chat_text(response: object) -> str:
    parsed: Final = ModelResponse.model_validate(response)
    choice: Final = OBJECT.validate_python(parsed.choices[0].model_dump())
    content: Final = OBJECT.validate_python(choice["message"])["content"]
    assert isinstance(content, str)
    return content


def messages_text(response: object) -> str:
    if isinstance(response, ModelResponse):
        return chat_text(response)
    content: Final = OBJECTS.validate_python(OBJECT.validate_python(response)["content"])
    text: Final = content[0]["text"]
    assert isinstance(text, str)
    return text


def ocr_text(response: object) -> str:
    return OCRResponse.model_validate(response).pages[0].markdown


def responses_text(response: object) -> str:
    return ResponsesAPIResponse.model_validate(response).output_text


def replace_chat(response: object, text: str) -> ModelResponse:
    parsed: Final = ModelResponse.model_validate(response)
    return ModelResponse.model_validate(
        {**parsed.model_dump(), "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}
    )


def replace_messages(response: object, text: str) -> dict[str, object]:
    return {**OBJECT.validate_python(response), "content": [{"type": "text", "text": text}]}


def replace_ocr(response: object, text: str) -> OCRResponse:
    parsed: Final = OCRResponse.model_validate(response)
    return parsed.model_copy(update={"pages": [parsed.pages[0].model_copy(update={"markdown": text})]})


def replace_responses(response: object, text: str) -> ResponsesAPIResponse:
    parsed: Final = ResponsesAPIResponse.model_validate(response)
    return ResponsesAPIResponse.model_validate(
        {
            **parsed.model_dump(),
            "output": [
                {
                    "type": "message",
                    "id": "msg_contract",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }
    )


def ocr_contract() -> CallbackRoute:
    return CallbackRoute(
        route=Route.OCR,
        arguments={"model": OCR_MODEL, "document": dict(OCR_DOCUMENT)},
        response=OCR_RESPONSE,
        call_types=("ocr", "aocr"),
        sync=litellm.ocr,
        asynchronous=litellm.aocr,
        text=ocr_text,
        replace=replace_ocr,
        expected_text="native OCR response",
    )


def messages_contract() -> CallbackRoute:
    return CallbackRoute(
        route=Route.MESSAGES,
        arguments={"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 64},
        response=MESSAGES_RESPONSE,
        call_types=("anthropic_messages", "anthropic_messages"),
        sync=litellm.anthropic.messages.create,
        asynchronous=litellm.anthropic.messages.acreate,
        text=messages_text,
        replace=replace_messages,
        expected_text="Hello from native Messages",
    )


def chat_contract() -> CallbackRoute:
    return CallbackRoute(
        route=Route.CHAT_COMPLETIONS,
        arguments={"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 64},
        response=MESSAGES_RESPONSE,
        call_types=("completion", "acompletion"),
        sync=litellm.completion,
        asynchronous=litellm.acompletion,
        text=chat_text,
        replace=replace_chat,
        expected_text="Hello from native Messages",
    )


def responses_contract() -> CallbackRoute:
    from tests.test_litellm_rust.support.requests import RESPONSES_MODEL, RESPONSES_RESPONSE

    return CallbackRoute(
        route=Route.RESPONSES,
        arguments={"model": RESPONSES_MODEL, "input": "hello", "max_output_tokens": 64},
        response=RESPONSES_RESPONSE,
        call_types=("responses", "aresponses"),
        sync=litellm.responses,
        asynchronous=litellm.aresponses,
        text=responses_text,
        replace=replace_responses,
        expected_text="native response",
    )
