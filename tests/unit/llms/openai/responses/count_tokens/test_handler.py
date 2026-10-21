import pytest

from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.count_tokens.handler import OpenAICountTokensHandler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "input_value", "expected_message"),
    [
        ("", "hello", "CountTokens processing error: model parameter is required"),
        ("gpt-4o", "", "CountTokens processing error: input parameter is required"),
        ("gpt-4o", [], "CountTokens processing error: input parameter is required"),
    ],
)
async def test_request_without_model_or_input_is_rejected_before_calling_openai(
    model: str, input_value: str | list[object], expected_message: str
) -> None:
    with pytest.raises(OpenAIError) as rejected:
        await OpenAICountTokensHandler().handle_count_tokens_request(
            model=model,
            input=input_value,
            api_key="sk-test",
        )

    assert (rejected.value.status_code, rejected.value.message) == (500, expected_message)
