import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import pytest

import litellm
from litellm import ModelResponse


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


def test_o1_supports_vision():
    """Test that o1 supports vision"""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    for k, v in litellm.model_cost.items():
        if k.startswith("o1") and v.get("litellm_provider") == "openai":
            assert v.get("supports_vision") is True, f"{k} does not support vision"


