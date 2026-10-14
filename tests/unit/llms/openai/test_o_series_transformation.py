from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import ModelResponse
from litellm.llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig
import os


@pytest.mark.parametrize(
    "model_name,expected",
    [
        # Valid O-series models
        ("o1", True),
        ("o3", True),
        ("o4-mini", True),
        ("o3-mini", True),
        # Valid O-series models with provider prefix
        ("openai/o1", True),
        ("openai/o3", True),
        ("openai/o4-mini", True),
        ("openai/o3-mini", True),
        # Non-O-series models
        ("gpt-4", False),
        ("gpt-3.5-turbo", False),
        ("claude-3-opus", False),
        # Non-O-series models with provider prefix
        ("openai/gpt-4", False),
        ("openai/gpt-3.5-turbo", False),
        ("anthropic/claude-3-opus", False),
        # Edge cases
        ("o", False),  # Too short
        ("o5", False),  # Not a valid O-series model
        ("o1-", False),  # Invalid suffix
        ("o3_", False),  # Invalid suffix
    ],
)
def test_is_model_o_series_model(model_name: str, expected: bool):
    """
    Test that is_model_o_series_model correctly identifies O-series models.

    Args:
        model_name: The model name to test
        expected: The expected result (True if it should be identified as an O-series model)
    """
    config = OpenAIOSeriesConfig()
    assert config.is_model_o_series_model(model_name) == expected, (
        f"Expected {model_name} to be {'an O-series model' if expected else 'not an O-series model'}"
    )


@pytest.mark.parametrize("model", ["o1"])
@pytest.mark.asyncio
async def test_o1_handle_system_role(model):
    """
    Tests that:
    - max_tokens is translated to 'max_completion_tokens'
    - role 'system' is translated to 'user'
    """
    from openai import AsyncOpenAI
    from litellm.utils import supports_system_messages

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    litellm.set_verbose = True

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model=model,
                max_tokens=10,
                messages=[{"role": "system", "content": "Be a good bot!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["model"] == model
        assert request_body["max_completion_tokens"] == 10
        if supports_system_messages(model, "openai"):
            assert request_body["messages"] == [
                {"role": "system", "content": "Be a good bot!"}
            ]
        else:
            assert request_body["messages"] == [
                {"role": "user", "content": "Be a good bot!"}
            ]


@pytest.mark.parametrize(
    "model, expected_tool_calling_support",
    [("o1", True)],
)
@pytest.mark.asyncio
async def test_o1_handle_tool_calling_optional_params(
    model, expected_tool_calling_support
):
    """
    Tests that:
    - max_tokens is translated to 'max_completion_tokens'
    - role 'system' is translated to 'user'
    """
    from litellm.utils import ProviderConfigManager
    from litellm.types.utils import LlmProviders

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    config = ProviderConfigManager.get_provider_chat_config(
        model=model, provider=LlmProviders.OPENAI
    )

    supported_params = config.get_supported_openai_params(model=model)

    assert expected_tool_calling_support == ("tools" in supported_params)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4", "gpt-4-0613"])
async def test_o1_max_completion_tokens(model: str):
    """
    Tests that:
    - max_completion_tokens is passed directly to OpenAI chat completion models
    """
    from openai import AsyncOpenAI

    litellm.set_verbose = True

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model=model,
                max_completion_tokens=10,
                messages=[{"role": "user", "content": "Hello!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["model"] == model
        assert request_body["max_completion_tokens"] == 10
        assert request_body["messages"] == [{"role": "user", "content": "Hello!"}]


def test_litellm_responses():
    """
    ensures that type of completion_tokens_details is correctly handled / returned
    """
    from litellm.types.utils import CompletionTokensDetails

    response = ModelResponse(
        usage={
            "completion_tokens": 436,
            "prompt_tokens": 14,
            "total_tokens": 450,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )

    print("response: ", response)

    assert isinstance(response.usage.completion_tokens_details, CompletionTokensDetails)
