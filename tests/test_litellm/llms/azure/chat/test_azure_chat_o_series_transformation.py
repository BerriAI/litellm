import json
import traceback
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

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


def test_azure_o_series_transform_request_flattens_top_level_anyof():
    """Regression test for LIT-6510: the o-series super() chain ends in
    OpenAIGPTConfig, whose flatten gate skips provider 'azure', so
    AzureOpenAIO1Config must flatten tool schema combinators itself."""
    tool = {
        "type": "function",
        "function": {
            "name": "automation_update",
            "description": "Update an automation",
            "parameters": {
                "type": "object",
                "anyOf": [
                    {
                        "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}},
                        "required": ["id", "enabled"],
                    },
                    {
                        "properties": {"id": {"type": "string"}, "schedule": {"type": "string"}},
                        "required": ["id", "schedule"],
                    },
                ],
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    }
    optional_params = {"tools": [tool]}

    request = AzureOpenAIO1Config().transform_request(
        model="o3-mini",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    parameters = request["tools"][0]["function"]["parameters"]
    assert "anyOf" not in parameters
    assert parameters["type"] == "object"
    assert set(parameters["properties"]) == {"id", "enabled", "schedule"}
    assert parameters["required"] == ["id"]
    assert "anyOf" in tool["function"]["parameters"]
    assert optional_params["tools"][0] is tool
