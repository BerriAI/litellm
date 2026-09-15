import json
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.bedrock.chat.openai_chat_completions_transformation import (
    AmazonBedrockOpenAIChatCompletionsConfig,
)
from litellm.llms.bedrock.common_utils import get_bedrock_chat_config
from litellm.llms.bedrock.messages.native_transformation import (
    AmazonBedrockNativeMessagesConfig,
    mantle_native_messages_config,
)
from litellm.llms.bedrock.responses.transformation import AmazonBedrockResponsesAPIConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, model_supports_native_endpoint


def _chat_response(url: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={  # mutable-ok: provider interface
            "id": "chat-test",
            "object": "chat.completion",
            "created": 1,
            "model": "native-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],  # mutable-ok: provider interface
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},  # mutable-ok: provider interface
        },
        request=httpx.Request("POST", url),
    )


def _messages_response(url: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={  # mutable-ok: provider interface
            "id": "message-test",
            "type": "message",
            "role": "assistant",
            "model": "native-model",
            "content": [{"type": "text", "text": "ok"}],  # mutable-ok: provider interface
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},  # mutable-ok: provider interface
        },
        request=httpx.Request("POST", url),
    )


def _responses_response(url: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={  # mutable-ok: provider interface
            "id": "response-test",
            "object": "response",
            "created_at": 1,
            "model": "native-model",
            "output": [],  # mutable-ok: provider interface
            "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},  # mutable-ok: provider interface
        },
        request=httpx.Request("POST", url),
    )


def test_native_endpoint_gate_reads_provider_model_cost(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/native-model",
        {"supported_endpoints": ["/v1/chat/completions"]},  # mutable-ok: provider interface
    )

    assert model_supports_native_endpoint(
        "/v1/chat/completions", "native-model", LlmProviders.BEDROCK
    )
    assert not model_supports_native_endpoint("/v1/messages", "native-model", LlmProviders.BEDROCK)


def test_native_endpoint_gate_uses_bedrock_base_model(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/anthropic.claude-3-7-sonnet",
        {"supported_endpoints": ["/v1/messages"]},  # mutable-ok: provider interface
    )

    assert model_supports_native_endpoint(
        "/v1/messages", "us.anthropic.claude-3-7-sonnet", LlmProviders.BEDROCK
    )


def test_bedrock_chat_route_prefers_explicit_converse(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/native-model",
        {"supported_endpoints": ["/v1/chat/completions"]},  # mutable-ok: provider interface
    )

    assert isinstance(get_bedrock_chat_config("native-model"), AmazonBedrockOpenAIChatCompletionsConfig)
    assert type(get_bedrock_chat_config("converse/native-model")).__name__ == "AmazonConverseConfig"


def test_bedrock_converse_catalog_model_prefers_declared_native_chat(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/anthropic.claude-sonnet-4-6",
        {"supported_endpoints": ["/v1/chat/completions"]},  # mutable-ok: provider interface
    )

    assert isinstance(
        get_bedrock_chat_config("anthropic.claude-sonnet-4-6"),
        AmazonBedrockOpenAIChatCompletionsConfig,
    )
    assert type(get_bedrock_chat_config("converse/anthropic.claude-sonnet-4-6")).__name__ == "AmazonConverseConfig"


def test_bedrock_messages_route_prefers_explicit_invoke(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/anthropic.claude-sonnet-4-6",
        {"supported_endpoints": ["/v1/messages"]},  # mutable-ok: provider interface
    )

    assert isinstance(
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model="anthropic.claude-sonnet-4-6",
            provider=LlmProviders.BEDROCK,
        ),
        AmazonBedrockNativeMessagesConfig,
    )
    assert type(
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model="invoke/anthropic.claude-sonnet-4-6",
            provider=LlmProviders.BEDROCK,
        )
    ).__name__ == "AmazonAnthropicClaudeMessagesConfig"


def test_native_chat_url_and_bearer_auth(monkeypatch):
    config = AmazonBedrockOpenAIChatCompletionsConfig()  # rebind-ok: test capture

    assert config.get_complete_url(
        None, None, "native-model", {"aws_region_name": "us-west-2"}, {}  # mutable-ok: provider interface
    ) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    validated_headers = config.validate_environment(
        {"content-type": "application/json"},  # mutable-ok: provider interface
        "native-model",
        [],  # mutable-ok: provider interface
        {},  # mutable-ok: provider interface
        {},  # mutable-ok: provider interface
    )
    headers, body = config.sign_request(
        validated_headers,
        {"aws_region_name": "us-west-2"},  # mutable-ok: provider interface
        {"model": "native-model"},  # mutable-ok: provider interface
        "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
    )
    assert headers["Authorization"] == "Bearer token"
    assert sum(key.lower() == "content-type" for key in headers) == 1
    assert body is not None


def test_native_chat_sigv4_auth(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    config = AmazonBedrockOpenAIChatCompletionsConfig()  # rebind-ok: test capture
    headers, _ = config.sign_request(
        {},  # mutable-ok: provider interface
        {  # mutable-ok: provider interface
            "aws_region_name": "us-west-2",
            "aws_access_key_id": "testing",
            "aws_secret_access_key": "testing",
        },
        {"model": "native-model"},  # mutable-ok: provider interface
        "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
    )

    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")


def test_native_messages_preserves_body_and_urls(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    config = AmazonBedrockNativeMessagesConfig()  # rebind-ok: test capture
    request = {  # mutable-ok: provider interface  # rebind-ok: test capture
        "model": "claude-native",
        "messages": [{"role": "user", "content": "hello"}],  # mutable-ok: provider interface
        "stream": True,
        "cache_control": {"ttl": "5m"},  # mutable-ok: provider interface
    }
    body = config.transform_anthropic_messages_request(  # rebind-ok: test capture
        model=request["model"],
        messages=request["messages"],
        anthropic_messages_optional_request_params={  # mutable-ok: provider interface
            "stream": request["stream"],
            "cache_control": request["cache_control"],
        },
        litellm_params={},  # mutable-ok: provider interface
        headers={},  # mutable-ok: provider interface
    )

    assert body == request
    assert "anthropic_version" not in body
    assert config.get_complete_url(None, None, "claude-native", {"aws_region_name": "us-west-2"}, {}) == (  # mutable-ok: provider interface
        "https://bedrock-runtime.us-west-2.amazonaws.com/anthropic/v1/messages"
    )
    assert mantle_native_messages_config().get_complete_url(
        None, None, "claude-native", {"aws_region_name": "us-west-2"}, {}  # mutable-ok: provider interface
    ) == "https://bedrock-mantle.us-west-2.api.aws/anthropic/v1/messages"
    headers, signed_body = config.sign_request(
        {},  # mutable-ok: provider interface
        {"aws_region_name": "us-west-2"},  # mutable-ok: provider interface
        body,  # mutable-ok: provider interface
        "https://bedrock-runtime.us-west-2.amazonaws.com/anthropic/v1/messages",
    )
    assert headers["Authorization"] == "Bearer token"
    assert sum(key.lower() == "content-type" for key in headers) == 1
    assert signed_body is not None


def test_native_messages_deduplicates_content_type_header(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    config = AmazonBedrockNativeMessagesConfig()  # rebind-ok: test capture
    headers, _ = config.validate_anthropic_messages_environment(
        headers={"content-type": "application/json"},  # mutable-ok: provider interface
        model="native-model",
        messages=[],  # mutable-ok: provider interface
        optional_params={},  # mutable-ok: provider interface
        litellm_params={},  # mutable-ok: provider interface
    )

    assert sum(key.lower() == "content-type" for key in headers) == 1
    assert headers["Content-Type"] == "application/json"


def test_native_messages_selection_is_not_cached(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/native-model",
        {"supported_endpoints": ["/v1/messages"]},  # mutable-ok: provider interface
    )
    assert isinstance(
        ProviderConfigManager.get_provider_anthropic_messages_config(
            "native-model", LlmProviders.BEDROCK
        ),
        AmazonBedrockNativeMessagesConfig,
    )

    monkeypatch.setitem(litellm.model_cost, "bedrock/native-model", {})  # mutable-ok: provider interface
    assert type(
        ProviderConfigManager.get_provider_anthropic_messages_config(
            "claude-native", LlmProviders.BEDROCK
        )
    ).__name__ == "AmazonAnthropicClaudeMessagesConfig"


def test_native_responses_selection_and_url(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/native-model",
        {"supported_endpoints": ["/v1/responses"]},  # mutable-ok: provider interface
    )
    config = ProviderConfigManager.get_provider_responses_api_config(  # rebind-ok: test capture
        LlmProviders.BEDROCK, "native-model"
    )

    assert isinstance(config, AmazonBedrockResponsesAPIConfig)
    assert config.get_complete_url(None, {"aws_region_name": "us-west-2"}).endswith(  # mutable-ok: provider interface
        "/openai/v1/responses"
    )
    validated_headers = config.validate_environment(
        {"content-type": "application/json"},  # mutable-ok: provider interface
        "native-model",
        {},  # mutable-ok: provider interface
    )
    assert sum(key.lower() == "content-type" for key in validated_headers) == 1

    monkeypatch.setitem(litellm.model_cost, "bedrock/native-model", {})  # mutable-ok: provider interface
    assert ProviderConfigManager.get_provider_responses_api_config(
        LlmProviders.BEDROCK, "native-model"
    ) is None


def test_completion_forwards_native_chat_body(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/native-model",
        {"supported_endpoints": ["/v1/chat/completions"]},  # mutable-ok: provider interface
    )
    requests: list[tuple[str, dict]] = []  # mutable-ok: provider interface  # rebind-ok: test capture

    def post(self, url, data=None, headers=None, **kwargs):  # kwargs-ok: HTTP handler compatibility
        requests.append((url, json.loads(data)))
        return _chat_response(url)

    with patch(  # test-quality-ok: captures the serialized provider request at the HTTP boundary
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post", post
    ):
        litellm.completion(
            model="bedrock/native-model",
            messages=[{"role": "user", "content": "hello"}],  # mutable-ok: provider interface
            api_base="https://bedrock-runtime.us-west-2.amazonaws.com",
            api_key="token",
        )

    assert requests == [  # mutable-ok: provider interface
        (
            "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
            {"model": "native-model", "messages": [{"role": "user", "content": "hello"}]},  # mutable-ok: provider interface
        )
    ]


def test_completion_forwards_declared_native_chat_for_converse_catalog_model(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/anthropic.claude-sonnet-4-6",
        {"supported_endpoints": ["/v1/chat/completions"]},  # mutable-ok: provider interface
    )
    requests: list[tuple[str, dict]] = []  # mutable-ok: provider interface  # rebind-ok: test capture

    def post(self, url, data=None, headers=None, **kwargs):  # kwargs-ok: HTTP handler compatibility
        requests.append((url, json.loads(data)))
        return _chat_response(url)

    with patch(  # test-quality-ok: captures the serialized provider request at the HTTP boundary
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post", post
    ):
        litellm.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],  # mutable-ok: provider interface
            api_base="https://bedrock-runtime.us-west-2.amazonaws.com",
            api_key="token",
        )

    assert requests[0][0].endswith("/openai/v1/chat/completions")


@pytest.mark.asyncio
async def test_anthropic_messages_forwards_native_body(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/native-model",
        {"supported_endpoints": ["/v1/messages"]},  # mutable-ok: provider interface
    )
    requests: list[tuple[str, dict]] = []  # mutable-ok: provider interface  # rebind-ok: test capture

    async def post(self, url, data=None, headers=None, **kwargs):  # kwargs-ok: HTTP handler compatibility
        requests.append((url, json.loads(data)))
        return _messages_response(url)

    with patch(  # test-quality-ok: captures the serialized provider request at the HTTP boundary
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new=post
    ):
        await litellm.anthropic_messages(
            model="bedrock/native-model",
            messages=[{"role": "user", "content": "hello"}],  # mutable-ok: provider interface
            max_tokens=10,
            api_base="https://bedrock-runtime.us-west-2.amazonaws.com",
            api_key="token",
        )

    assert requests == [  # mutable-ok: provider interface
        (
            "https://bedrock-runtime.us-west-2.amazonaws.com/anthropic/v1/messages",
            {  # mutable-ok: provider interface
                "max_tokens": 10,
                "stream": False,
                "model": "native-model",
                "messages": [{"role": "user", "content": "hello"}],  # mutable-ok: provider interface
            },
        )
    ]


def test_responses_forwards_native_body(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/native-model",
        {"supported_endpoints": ["/v1/responses"]},  # mutable-ok: provider interface
    )
    requests: list[tuple[str, dict]] = []  # mutable-ok: provider interface  # rebind-ok: test capture

    def post(self, url, data=None, headers=None, **kwargs):  # kwargs-ok: HTTP handler compatibility
        requests.append((url, json.loads(data)))
        return _responses_response(url)

    with patch(  # test-quality-ok: captures the serialized provider request at the HTTP boundary
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post", post
    ):
        litellm.responses(
            model="bedrock/native-model",
            input="hello",
            instructions="be brief",
            api_base="https://bedrock-runtime.us-west-2.amazonaws.com",
            api_key="token",
        )

    assert requests == [  # mutable-ok: provider interface
        (
            "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/responses",
            {  # mutable-ok: provider interface
                "model": "native-model",
                "input": "hello",
                "instructions": "be brief",
            },
        )
    ]
