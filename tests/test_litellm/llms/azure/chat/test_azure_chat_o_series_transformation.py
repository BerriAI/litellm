import json
import os
import sys
import traceback
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.azure.chat.o_series_transformation import AzureOpenAIO1Config


@pytest.mark.asyncio
async def test_azure_chat_o_series_transformation():
    provider_config = AzureOpenAIO1Config()
    model = "o_series/web-interface-o1-mini"
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    optional_params = {}
    litellm_params = {}
    headers = {}

    response = await provider_config.async_transform_request(
        model, messages, optional_params, litellm_params, headers
    )
    print(response)
    assert response["model"] == "web-interface-o1-mini"


@pytest.mark.parametrize("model", ["o1", "o3-mini"])
def test_azure_o_series_tool_choice_required_gated_by_api_version(model):
    """
    tool_choice='required' is not supported by Azure on api_version<=2024-05-01.

    Azure o-series deployments must honor the same api_version gating as the
    non-o-series Azure path instead of silently forwarding 'required'.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.get_optional_params(
            model=model,
            custom_llm_provider="azure",
            tool_choice="required",
            tools=tools,
            api_version="2024-05-01-preview",
            drop_params=False,
        )

    # newer api_version supports it
    params = litellm.get_optional_params(
        model=model,
        custom_llm_provider="azure",
        tool_choice="required",
        tools=tools,
        api_version="2025-01-01-preview",
        drop_params=False,
    )
    assert params["tool_choice"] == "required"


@pytest.mark.parametrize("model", ["o1", "o3-mini"])
def test_azure_o_series_response_format_falls_back_to_tools_on_old_api_version(model):
    """
    On api_versions that predate native json_schema support, Azure o-series should
    convert response_format into a tool call, matching the non-o-series Azure path.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "r",
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
            "strict": True,
        },
    }
    params = litellm.get_optional_params(
        model=model,
        custom_llm_provider="azure",
        response_format=response_format,
        api_version="2024-02-01",
        drop_params=False,
    )
    assert "response_format" not in params
    assert "tools" in params


def test_azure_o_series_still_maps_max_tokens():
    """The o-series max_tokens -> max_completion_tokens translation must be preserved."""
    params = litellm.get_optional_params(
        model="o3-mini",
        custom_llm_provider="azure",
        max_tokens=64,
        api_version="2025-01-01-preview",
        drop_params=False,
    )
    assert params["max_completion_tokens"] == 64
    assert "max_tokens" not in params


def test_azure_o_series_without_gated_params_is_unchanged():
    """
    When no api_version-gated params (tool_choice/response_format) are passed,
    the o-series mapping should behave exactly like the OpenAI o-series path.
    """
    params = litellm.get_optional_params(
        model="o3-mini",
        custom_llm_provider="azure",
        max_tokens=64,
        reasoning_effort="high",
        api_version="2024-02-01",
        drop_params=False,
    )
    assert params["max_completion_tokens"] == 64
    assert params["reasoning_effort"] == "high"
    assert "tools" not in params
