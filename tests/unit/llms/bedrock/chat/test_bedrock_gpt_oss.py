import json
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def test_function_calling_request_body_gpt_oss():
    client = HTTPHandler()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a city",
                "parameters": {
                    "$id": "https://some/internal/name",
                    "$schema": "https://json-schema.org/draft-07/schema",
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to get the weather for",
                        }
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    response = httpx.Response(
        200,
        json={
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
        },
        request=httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
    )
    with patch.object(client, "post", new=Mock(return_value=response)) as mock_post:
        litellm.completion(
            model="bedrock/converse/openai.gpt-oss-20b-1:0",
            messages=[{"role": "user", "content": "How is the weather in Mumbai?"}],
            tools=tools,
            aws_region_name="us-west-2",
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            client=client,
        )

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"].endswith("/model/openai.gpt-oss-20b-1%3A0/converse"), call_kwargs["url"]

    request_body = json.loads(call_kwargs["data"])

    assert "toolConfig" in request_body
    tool_specs = request_body["toolConfig"]["tools"]
    assert len(tool_specs) == 1
    tool_spec = tool_specs[0]["toolSpec"]
    assert tool_spec["name"] == "get_weather"
    assert tool_spec["description"] == "Get the weather in a city"

    input_schema = tool_spec["inputSchema"]["json"]
    assert input_schema["type"] == "object"
    assert input_schema["required"] == ["city"]
    assert input_schema["properties"]["city"]["type"] == "string"

    for stripped_field in ("$id", "$schema", "additionalProperties", "strict"):
        assert stripped_field not in input_schema, f"{stripped_field} should be stripped before hitting Bedrock"

    assert request_body["messages"][0]["role"] == "user"
    assert request_body["messages"][0]["content"][0]["text"] == "How is the weather in Mumbai?"


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/openai.gpt-oss-20b-1:0",
        "bedrock/openai.gpt-oss-120b-1:0",
    ],
)
def test_reasoning_effort_transformation_gpt_oss(model):
    config = AmazonConverseConfig()
    non_default_params = {"reasoning_effort": "low"}
    optional_params = {}

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )

    assert "reasoning_effort" in result
    assert result["reasoning_effort"] == "low"
    assert "thinking" not in result
