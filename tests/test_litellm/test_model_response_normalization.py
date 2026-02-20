import warnings

import pytest

from litellm.types.utils import Choices, Delta, Message, ModelResponse, StreamingChoices


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


def test_modelresponse_model_dump_json_no_pydantic_warnings() -> None:
    """model_dump_json() bypasses the Python model_dump() override and uses
    Pydantic's C-level serializer directly.  The Union[Choices, StreamingChoices]
    field previously triggered PydanticSerializationUnexpectedValue warnings via
    this path."""
    response = ModelResponse(
        model="test-model",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="hello", role="assistant"),
            )
        ],
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _ = response.model_dump_json()
        _ = response.model_dump()
        _ = response.model_dump(exclude_none=True)

    pydantic_warnings = [
        w
        for w in captured
        if "PydanticSerializationUnexpectedValue" in str(w.message)
        or "Pydantic serializer warnings" in str(w.message)
    ]
    assert pydantic_warnings == [], (
        f"Unexpected Pydantic serialization warnings: {pydantic_warnings}"
    )


def test_streaming_modelresponse_no_pydantic_warnings() -> None:
    """Streaming responses use StreamingChoices in the Union field and should
    also serialize without warnings."""
    response = ModelResponse(
        model="test-model",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content="hello", role="assistant"),
            )
        ],
        stream=True,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _ = response.model_dump_json()
        _ = response.model_dump()

    pydantic_warnings = [
        w
        for w in captured
        if "PydanticSerializationUnexpectedValue" in str(w.message)
        or "Pydantic serializer warnings" in str(w.message)
    ]
    assert pydantic_warnings == [], (
        f"Unexpected Pydantic serialization warnings: {pydantic_warnings}"
    )
