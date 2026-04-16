import warnings

import pytest

from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    _coerce_provider_specific_fields,
)


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
    """model_dump_json() and model_dump() should not trigger any Pydantic
    serialization warnings now that choices is List[Choices] (no Union)."""
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


def test_streaming_modelresponsestream_no_pydantic_warnings() -> None:
    """Streaming responses use ModelResponseStream with List[StreamingChoices]
    and should serialize without warnings."""
    response = ModelResponseStream(
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content="hello", role="assistant"),
            )
        ],
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


def test_coerce_provider_specific_fields_converts_basemodel_to_dict() -> None:
    msg = Message(content="test", role="assistant")
    result = _coerce_provider_specific_fields({"message": msg})
    assert isinstance(result["message"], dict)
    assert result["message"]["content"] == "test"
    assert result["message"]["role"] == "assistant"


def test_coerce_provider_specific_fields_handles_nested_structures() -> None:
    msg = Message(content="nested", role="user")
    result = _coerce_provider_specific_fields({
        "items": [msg, {"plain": "dict"}],
        "nested_dict": {"inner": msg},
        "scalar": 42,
    })
    assert isinstance(result["items"][0], dict)
    assert result["items"][0]["content"] == "nested"
    assert result["items"][1] == {"plain": "dict"}
    assert isinstance(result["nested_dict"]["inner"], dict)
    assert result["scalar"] == 42


def test_coerce_provider_specific_fields_passthrough_plain_values() -> None:
    result = _coerce_provider_specific_fields({
        "text": "hello",
        "count": 5,
        "flag": True,
        "nested": {"key": "value"},
    })
    assert result == {
        "text": "hello",
        "count": 5,
        "flag": True,
        "nested": {"key": "value"},
    }


def test_streaming_provider_specific_fields_with_basemodel_no_warnings() -> None:
    msg = Message(content="provider-data", role="assistant")
    response = ModelResponseStream(
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="hello", role="assistant"),
            )
        ],
        provider_specific_fields={"message": msg},
    )

    assert isinstance(response.provider_specific_fields["message"], dict)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _ = response.model_dump()
        _ = response.model_dump_json()

    pydantic_warnings = [
        w
        for w in captured
        if "PydanticSerializationUnexpectedValue" in str(w.message)
        or "Pydantic serializer warnings" in str(w.message)
    ]
    assert pydantic_warnings == [], (
        f"Unexpected Pydantic serialization warnings: {pydantic_warnings}"
    )


def test_choices_provider_specific_fields_with_basemodel_no_warnings() -> None:
    msg = Message(content="extra-data", role="assistant")
    choice = Choices(
        finish_reason="stop",
        index=0,
        message=Message(content="main", role="assistant"),
        provider_specific_fields={"extra": msg},
    )

    assert isinstance(choice.provider_specific_fields["extra"], dict)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        response = ModelResponse(model="test", choices=[choice])
        _ = response.model_dump()
        _ = response.model_dump_json()

    pydantic_warnings = [
        w
        for w in captured
        if "PydanticSerializationUnexpectedValue" in str(w.message)
        or "Pydantic serializer warnings" in str(w.message)
    ]
    assert pydantic_warnings == [], (
        f"Unexpected Pydantic serialization warnings: {pydantic_warnings}"
    )


def test_delta_provider_specific_fields_with_basemodel_coerced() -> None:
    msg = Message(content="delta-data", role="assistant")
    delta = Delta(
        content="hi",
        role="assistant",
        provider_specific_fields={"msg": msg},
    )
    assert isinstance(delta.provider_specific_fields["msg"], dict)
    assert delta.provider_specific_fields["msg"]["content"] == "delta-data"
