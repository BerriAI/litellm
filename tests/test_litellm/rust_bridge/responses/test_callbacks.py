from types import MappingProxyType
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.rust_bridge.responses.callbacks import arguments, response
from litellm.rust_bridge.responses.entrypoints import LiteLLMResponsesRequest
from litellm.types.llms.openai import ResponsesAPIResponse


def test_response_validates_into_the_public_responses_model() -> None:
    built: Final = response(
        MappingProxyType(
            {
                "id": "resp_native",
                "object": "response",
                "created_at": 1,
                "model": "gpt-4o",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_native",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "native", "annotations": []}],
                    }
                ],
            }
        )
    )

    assert isinstance(built, ResponsesAPIResponse)
    assert built.id == "resp_native"
    assert built.output[0].content[0].text == "native"


def test_response_rejects_a_payload_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        response(MappingProxyType({"object": "response"}))


def test_arguments_are_the_public_kwargs_view() -> None:
    kwargs: Final = MappingProxyType({"litellm_metadata": {"user_id": "u"}})
    request: Final = LiteLLMResponsesRequest(
        model="gpt-4o",
        input="hi",
        stream=None,
        api_key=None,
        api_base=None,
        custom_llm_provider="openai",
        extra_headers=None,
        kwargs=kwargs,
    )

    assert arguments(request) is kwargs
