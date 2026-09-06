from __future__ import annotations

import pytest

import litellm
from litellm.rust_bridge import chat_completions as bridge
from litellm.rust_bridge.request import (
    NativeChatCompletionsRequest,
    NativeRequestContext,
    NativeRequestOptions,
)


@pytest.fixture(autouse=True)
def native_bridge(monkeypatch):
    monkeypatch.setenv("LITELLM_RUST", "1")
    yield
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None)


def test_native_bedrock_receives_explicit_auth_and_endpoint():
    requests = []

    def native(
        request: NativeChatCompletionsRequest,
        *,
        options: NativeRequestOptions,
        context: NativeRequestContext,
        callback_adapter=None,
    ):
        requests.append((request, options))
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "native"}, "finish_reason": "stop"}]
        }

    bridge.set_rust_chat_completions(chat_completions=native)
    response = litellm.completion(
        model="bedrock/anthropic.claude-sonnet-4-5-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        aws_access_key_id="explicit-id",
        aws_secret_access_key="explicit-secret",
        aws_session_token="explicit-token",
        aws_region_name="us-east-1",
        api_base="https://example.test",
        max_tokens=7,
    )
    assert response.choices[0].message.content == "native"
    assert len(requests) == 1
    assert requests[0][1].bedrock == bridge.NativeBedrockOptions(
        aws_access_key_id="explicit-id",
        aws_secret_access_key="explicit-secret",
        aws_session_token="explicit-token",
        aws_region_name="us-east-1",
    )
    assert requests[0][1].api_base == "https://example.test"


@pytest.mark.parametrize("through_environment", [False, True])
def test_native_bedrock_preserves_bearer_auth(monkeypatch, through_environment):
    requests = []

    def native(request, *, options, context, callback_adapter=None):
        requests.append((request, options))
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "native"}, "finish_reason": "stop"}]
        }

    bridge.set_rust_chat_completions(chat_completions=native)
    if through_environment:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token")
    litellm.completion(
        model="bedrock/anthropic.claude-sonnet-4-5-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        api_key=None if through_environment else "bedrock-token",
        max_tokens=7,
    )
    assert len(requests) == 1
    assert requests[0][1].api_key == (None if through_environment else "bedrock-token")
