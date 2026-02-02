import warnings

import pytest

from litellm.types.utils import Choices, Message, ModelResponse


def test_modelresponse_normalizes_openai_base_models() -> None:
    # OpenAI SDK returns Pydantic BaseModel objects for message/choice.
    # LiteLLM should normalize these into its own internal `Message` / `Choices` types.
    from openai.types.chat.chat_completion import Choice as OpenAIChoice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    message = ChatCompletionMessage(role="assistant", content="hi")
    choice = OpenAIChoice(finish_reason="stop", index=0, message=message, logprobs=None)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        response = ModelResponse(model="gpt-4o-mini", choices=[choice])
        _ = response.model_dump()

    assert isinstance(response.choices[0], Choices)
    assert isinstance(response.choices[0].message, Message)

    assert not any(
        "Pydantic serializer warnings" in str(w.message)
        for w in captured
        if isinstance(w.message, Warning)
    )


def test_modelresponse_serialization_avoids_pydantic_warnings() -> None:
    pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion as OpenAIChatCompletion

    openai_completion = OpenAIChatCompletion(
        id="test-1",
        created=1719868600,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"},
                "logprobs": None,
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        response = ModelResponse(**openai_completion.model_dump())
        _ = response.model_dump(exclude_none=True)

    assert not any(
        "PydanticSerializationUnexpectedValue" in str(w.message)
        or "Pydantic serializer warnings" in str(w.message)
        for w in captured
    )
