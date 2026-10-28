"""
Unit tests for DashScope's Anthropic-compatible Messages endpoint.
"""

from typing import Final

import pytest
import respx

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.dashscope.messages.transformation import (
    DashScopeAnthropicMessagesConfig,
)

WORKSPACE: Final = "https://llm-abc123.cn-beijing.maas.aliyuncs.com"
INTL: Final = "https://dashscope-intl.aliyuncs.com"
CODING_PLAN: Final = "https://coding-intl.dashscope.aliyuncs.com"
CHAT_COMPLETION: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1792227137,
    "model": "qwen3.7-plus",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
}
MESSAGE: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "qwen3.7-plus",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 9, "output_tokens": 1},
}


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (
            None,
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com/apps/anthropic",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            WORKSPACE,
            f"{WORKSPACE}/apps/anthropic/v1/messages",
        ),
        (
            f"{WORKSPACE}/compatible-mode/v1",
            f"{WORKSPACE}/apps/anthropic/v1/messages",
        ),
        (
            "https://gateway.example.com/anthropic",
            "https://gateway.example.com/anthropic/v1/messages",
        ),
    ],
)
def test_get_complete_url(api_base: str | None, expected: str):
    config: Final = DashScopeAnthropicMessagesConfig()

    url: Final = config.get_complete_url(
        api_base=api_base,
        api_key="sk-test",
        model="qwen3.7-plus",
        optional_params={},
        litellm_params={},
    )

    assert url == expected


@pytest.mark.parametrize("dashscope_api_key", [None, "   "])
def test_missing_dashscope_key_does_not_fall_back_to_anthropic(
    monkeypatch: pytest.MonkeyPatch, dashscope_api_key: str | None
):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    if dashscope_api_key is not None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", dashscope_api_key)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setattr(litellm, "api_key", None)
    headers: dict[str, str] = {}

    with pytest.raises(AuthenticationError) as error:
        DashScopeAnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers=headers,
            model="qwen3.7-plus",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=WORKSPACE,
        )

    assert "DashScope" in str(error.value)
    assert "sk-ant-secret" not in str(error.value)
    assert {name.lower() for name in headers}.isdisjoint({"x-api-key", "authorization"})


def test_caller_auth_header_is_honored_without_a_dashscope_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    headers: Final = {"x-api-key": "caller-key"}

    resolved_headers, _ = DashScopeAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers=headers,
        model="qwen3.7-plus",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=WORKSPACE,
    )

    assert resolved_headers["x-api-key"] == "caller-key"


def test_caller_authorization_header_is_honored_without_a_dashscope_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    headers: Final = {"Authorization": "Bearer caller-token"}

    resolved_headers, _ = DashScopeAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers=headers,
        model="qwen3.7-plus",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=WORKSPACE,
    )

    assert resolved_headers["Authorization"] == "Bearer caller-token"


def test_dashscope_key_wins_over_an_anthropic_environment_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    resolved_headers, _ = DashScopeAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={},
        model="qwen3.7-plus",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=WORKSPACE,
    )

    header_values: Final = " ".join(resolved_headers.values())
    assert "sk-dashscope" in header_values
    assert "sk-ant-secret" not in header_values


def test_dashscope_prefix_reaches_the_provider():
    model, provider, _, _ = litellm.get_llm_provider(
        "dashscope/qwen3.7-plus",
        api_base=WORKSPACE,
    )

    assert provider == "dashscope"
    assert model == "qwen3.7-plus"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        (INTL, f"{INTL}/apps/anthropic/v1/messages"),
        (f"{INTL}/apps/anthropic", f"{INTL}/apps/anthropic/v1/messages"),
        (WORKSPACE, f"{WORKSPACE}/apps/anthropic/v1/messages"),
        (f"{CODING_PLAN}/apps/anthropic", f"{CODING_PLAN}/apps/anthropic/v1/messages"),
        (None, "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        (f"{INTL}/compatible-mode/v1", f"{INTL}/compatible-mode/v1/chat/completions"),
        (f"{CODING_PLAN}/v1", f"{CODING_PLAN}/v1/chat/completions"),
        ("https://gateway.example.com/anthropic", "https://gateway.example.com/anthropic/chat/completions"),
    ],
)
async def test_only_a_domain_root_or_anthropic_base_leaves_the_chat_bridge(
    monkeypatch: pytest.MonkeyPatch, api_base: str | None, expected_url: str
):
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock(assert_all_called=False) as upstream:
        upstream.route(path__regex=r".*/chat/completions$").respond(200, json=CHAT_COMPLETION)
        upstream.route(path__regex=r".*/v1/messages$").respond(200, json=MESSAGE)

        await litellm.anthropic.messages.acreate(
            model="dashscope/qwen3.7-plus",
            messages=[{"role": "user", "content": "say ok"}],
            max_tokens=16,
            api_key="sk-dashscope",
            api_base=api_base,
        )

        assert [str(call.request.url) for call in upstream.calls] == [expected_url]
