import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.llms.hosted_vllm.count_tokens.token_counter import (
    TOKENIZER_TYPE,
    HostedVLLMTokenCounter,
    chat_request_for_tokenize,
    is_anthropic_shaped,
    tokenize_url,
)
from litellm.types.utils import TokenCountResponse

_DEPLOYMENT: Final = {
    "litellm_params": {
        "model": "hosted_vllm/qwen",
        "api_base": "http://vllm:8000/v1",
        "api_key": "secret",
    }
}
_ANTHROPIC_MESSAGES: Final = [
    {"role": "user", "content": "list the files"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "I should call ls.", "signature": None},
            {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {"path": "."}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.py\nb.py"}],
    },
]
_ANTHROPIC_TOOLS: Final = [
    {
        "name": "ls",
        "description": "List a directory",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
]


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("http://vllm:8000/v1", "http://vllm:8000/tokenize"),
        ("http://vllm:8000/v1/", "http://vllm:8000/tokenize"),
        ("http://vllm:8000", "http://vllm:8000/tokenize"),
        ("https://gw.example/vllm/v1", "https://gw.example/vllm/tokenize"),
    ],
)
def test_tokenize_url_strips_the_v1_prefix(api_base: str, expected: str) -> None:
    assert tokenize_url(api_base) == expected


def test_is_anthropic_shaped_by_system_blocks_and_roles() -> None:
    assert is_anthropic_shaped([{"role": "user", "content": "hi"}], system="be brief")
    assert is_anthropic_shaped(_ANTHROPIC_MESSAGES, system=None)
    assert not is_anthropic_shaped([{"role": "user", "content": "hi"}], system=None)
    assert not is_anthropic_shaped(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], system=None
    )
    assert not is_anthropic_shaped(
        [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function"}]}],
        system=None,
    )


def test_chat_request_for_tokenize_translates_anthropic_input_and_drops_thinking() -> None:
    body: Final = chat_request_for_tokenize(
        "qwen", _ANTHROPIC_MESSAGES, _ANTHROPIC_TOOLS, system="be brief", request_model="claude-opus-5"
    )

    assert body["model"] == "qwen"
    assert body["add_generation_prompt"] is True
    messages: Final = body["messages"]
    assert isinstance(messages, list)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assistant: Final = messages[2]
    # the provider transform keeps string reasoning_content (what the chat path sends) and
    # drops thinking_blocks, so the count matches the rendered prompt
    assert assistant["reasoning_content"] == "I should call ls."
    assert "thinking_blocks" not in assistant
    assert assistant["tool_calls"][0]["function"]["name"] == "ls"
    assert "thinking_blocks" not in json.dumps(messages)
    tools: Final = body["tools"]
    assert isinstance(tools, list)
    assert tools[0]["type"] == "function" and tools[0]["function"]["name"] == "ls"


def test_chat_request_for_tokenize_passes_openai_input_through() -> None:
    openai_messages: Final = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    body: Final = chat_request_for_tokenize("qwen", openai_messages, None, system=None)

    assert body["messages"] == openai_messages
    assert "tools" not in body


def _client_returning(payload: object, status_code: int = 200) -> MagicMock:
    response: Final = MagicMock()
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=MagicMock(status_code=status_code)
        )
    client: Final = MagicMock()
    client.post = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_count_tokens_posts_the_translated_request_to_tokenize() -> None:
    client: Final = _client_returning({"count": 4242, "max_model_len": 500000, "tokens": [1, 2, 3]})
    with patch("litellm.llms.hosted_vllm.count_tokens.token_counter.get_async_httpx_client", return_value=client):
        result: Final = await HostedVLLMTokenCounter().count_tokens(
            model_to_use="qwen",
            messages=list(_ANTHROPIC_MESSAGES),
            contents=None,
            deployment=dict(_DEPLOYMENT),
            request_model="claude-opus-5",
            tools=list(_ANTHROPIC_TOOLS),
            system="be brief",
        )

    assert result == TokenCountResponse(
        total_tokens=4242,
        request_model="claude-opus-5",
        model_used="qwen",
        tokenizer_type=TOKENIZER_TYPE,
        original_response={"count": 4242, "max_model_len": 500000},
    )
    call: Final = client.post.await_args
    assert call.args == ("http://vllm:8000/tokenize",)
    assert call.kwargs["headers"] == {"Authorization": "Bearer secret"}
    sent: Final = call.kwargs["json"]
    assert sent["model"] == "qwen" and sent["add_generation_prompt"] is True
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "tool"]
    assert sent["tools"][0]["function"]["name"] == "ls"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client",
    [
        _client_returning({"error": "nope"}, status_code=500),
        _client_returning({"max_model_len": 1}),
        _client_returning("not json"),
    ],
)
async def test_count_tokens_falls_back_to_local_counting_on_failure(client: MagicMock) -> None:
    with patch("litellm.llms.hosted_vllm.count_tokens.token_counter.get_async_httpx_client", return_value=client):
        result: Final = await HostedVLLMTokenCounter().count_tokens(
            model_to_use="qwen",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment=dict(_DEPLOYMENT),
        )

    assert result is None


@pytest.mark.asyncio
async def test_count_tokens_without_api_base_or_messages_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    counter: Final = HostedVLLMTokenCounter()

    assert await counter.count_tokens(model_to_use="qwen", messages=[], contents=None) is None
    assert (
        await counter.count_tokens(
            model_to_use="qwen", messages=[{"role": "user", "content": "hi"}], contents=None, deployment={}
        )
        is None
    )


def test_should_use_token_counting_api_only_for_hosted_vllm() -> None:
    counter: Final = HostedVLLMTokenCounter()

    assert counter.should_use_token_counting_api("hosted_vllm")
    assert not counter.should_use_token_counting_api("vllm")
    assert not counter.should_use_token_counting_api("openai")


def test_chat_request_for_tokenize_does_not_mutate_the_caller_messages() -> None:
    """The transform pops reasoning fields in place; counting must not change the request."""
    openai_messages: Final = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "reasoning_content": {"not": "a string"},
            "thinking_blocks": [{"type": "thinking", "thinking": "hm", "signature": "s"}],
        },
    ]
    before: Final = json.dumps(openai_messages, sort_keys=True)

    chat_request_for_tokenize("qwen", openai_messages, None, system=None)

    assert json.dumps(openai_messages, sort_keys=True) == before


@pytest.mark.asyncio
async def test_count_tokens_builds_the_body_off_the_event_loop() -> None:
    """The sync transform can fetch user-supplied file URLs; it must not run on the loop thread."""
    import threading

    loop_thread: Final = threading.current_thread()
    seen: Final[list[threading.Thread]] = []

    def recording_builder(*args: object, **kwargs: object) -> dict[str, object]:
        seen.append(threading.current_thread())
        return {"model": "qwen", "messages": [], "add_generation_prompt": True}

    client: Final = _client_returning({"count": 7, "max_model_len": 1000})
    with (
        patch("litellm.llms.hosted_vllm.count_tokens.token_counter.get_async_httpx_client", return_value=client),
        patch(
            "litellm.llms.hosted_vllm.count_tokens.token_counter.chat_request_for_tokenize",
            side_effect=recording_builder,
        ),
    ):
        result: Final = await HostedVLLMTokenCounter().count_tokens(
            model_to_use="qwen",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment=dict(_DEPLOYMENT),
            request_model="claude-opus-5",
            tools=None,
            system=None,
        )

    assert result is not None and result.total_tokens == 7
    assert seen and seen[0] is not loop_thread
