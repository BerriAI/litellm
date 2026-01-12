import warnings

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
