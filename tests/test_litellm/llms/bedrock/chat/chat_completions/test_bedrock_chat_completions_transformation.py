"""Native Bedrock Runtime Chat Completions: Grok stays on /openai/v1/chat/completions."""

import json
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.bedrock.chat.chat_completions.transformation import (
    AmazonBedrockRuntimeChatCompletionsConfig,
)
from litellm.llms.bedrock.common_utils import (
    BedrockModelInfo,
    get_bedrock_chat_config,
    uses_bedrock_runtime_chat_completions,
)


@pytest.fixture
def local_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    try:
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.get_model_info.cache_clear()
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


@pytest.mark.parametrize(
    "model",
    [
        "us.xai.grok-4.6",
        "global.xai.grok-4.6",
        "us-gov.xai.grok-4.6",
        "bedrock/us.xai.grok-4.6",
    ],
)
def test_grok_runtime_models_use_chat_completions_route(local_cost_map, model):
    assert uses_bedrock_runtime_chat_completions(model) is True
    assert BedrockModelInfo.get_bedrock_route(model) == "chat_completions"
    assert isinstance(get_bedrock_chat_config(model), AmazonBedrockRuntimeChatCompletionsConfig)


def test_explicit_converse_prefix_still_uses_converse(local_cost_map):
    assert BedrockModelInfo.get_bedrock_route("bedrock/converse/us.xai.grok-4.6") == "converse"
    assert BedrockModelInfo.get_bedrock_route("converse/us.xai.grok-4.6") == "converse"


def test_claude_stays_on_converse(local_cost_map):
    assert uses_bedrock_runtime_chat_completions("us.anthropic.claude-3-sonnet-20240229-v1:0") is False
    assert BedrockModelInfo.get_bedrock_route("us.anthropic.claude-3-sonnet-20240229-v1:0") == "converse"


def test_flag_absent_means_no_chat_completions_route(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", {"us.xai.grok-4.6": {"litellm_provider": "bedrock_converse"}})
    assert uses_bedrock_runtime_chat_completions("us.xai.grok-4.6") is False


def test_complete_url_is_runtime_openai_chat_completions(monkeypatch):
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    url = cfg.get_complete_url(
        api_base=None,
        api_key=None,
        model="us.xai.grok-4.6",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions"


def test_complete_url_appends_to_openai_v1_base():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    url = cfg.get_complete_url(
        api_base="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
        api_key=None,
        model="us.xai.grok-4.6",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"


def test_transform_request_is_openai_chat_body_not_converse():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    body = cfg.transform_request(
        model="bedrock/us.xai.grok-4.6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"temperature": 0.2, "aws_region_name": "us-east-1"},
        litellm_params={},
        headers={},
    )
    assert body["model"] == "us.xai.grok-4.6"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["temperature"] == 0.2
    assert "aws_region_name" not in body
    assert "inferenceConfig" not in body
    assert "messages" in body


def test_completion_posts_runtime_chat_completions(local_cost_map, monkeypatch):
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")

    requests: list[dict] = []

    def mock_post(self, url, data=None, json=None, headers=None, **kwargs):
        requests.append({"url": url, "data": data, "json": json, "headers": headers or {}})
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1733529600,
                "model": "us.xai.grok-4.6",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            request=httpx.Request("POST", url),
        )

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", mock_post):
        response = litellm.completion(
            model="us.xai.grok-4.6",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert response.choices[0].message.content == "ok"
    assert len(requests) == 1
    assert requests[0]["url"] == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    raw = requests[0]["data"]
    body = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else (requests[0]["json"] or {})
    assert body["model"] == "us.xai.grok-4.6"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert "inferenceConfig" not in body
