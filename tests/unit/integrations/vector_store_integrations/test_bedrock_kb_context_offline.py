import itertools
import json
from typing import Final, TypedDict

from typing_extensions import ReadOnly

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
import litellm.proxy.proxy_server as proxy_server
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import VectorStorePreCallHook
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.vector_stores.vector_store_registry import LiteLLM_ManagedVectorStore, VectorStoreRegistry

_KB_ID: Final = "T37J8R4WTM"
_KB_URL: Final = f"https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases/{_KB_ID}/retrieve"
_ANTHROPIC_URL: Final = "https://api.anthropic.com/v1/messages"
_OPENAI_URL: Final = "https://api.openai.com/v1/chat/completions"
_KB_TEXT: Final = "LiteLLM is a library that simplifies LLM API access"
_PREFIX: Final = VectorStorePreCallHook.CONTENT_PREFIX_STRING
_BODY: Final = TypeAdapter(dict[str, object])

_ANTHROPIC_MESSAGE: Final = {
    "id": "msg_kb",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "LiteLLM simplifies LLM access."}],
    "model": "claude-haiku-4-5-20251001",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 100, "output_tokens": 50},
}
_OPENAI_COMPLETION: Final = {
    "id": "chatcmpl-kb",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-5-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}
_ANTHROPIC_STREAM: Final = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_kb_stream","type":"message",'
    '"role":"assistant","content":[],"model":"claude-haiku-4-5-20251001","stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"LiteLLM"}}\n\n'
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
    '"usage":{"output_tokens":2}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


class _ChatMessage(TypedDict):
    role: ReadOnly[str]
    content: ReadOnly[str]


@pytest.fixture(autouse=True)
def _knowledge_base(monkeypatch: pytest.MonkeyPatch, fake_provider_credentials: None) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    monkeypatch.setattr(
        litellm,
        "vector_store_registry",
        VectorStoreRegistry(
            vector_stores=[LiteLLM_ManagedVectorStore(vector_store_id=_KB_ID, custom_llm_provider="bedrock")]
        ),
        raising=False,
    )


def _kb_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(_KB_URL).mock(
        return_value=httpx.Response(
            200,
            json={"retrievalResults": [{"content": {"text": _KB_TEXT, "type": "TEXT"}, "score": 0.9, "metadata": {}}]},
        )
    )


def _sent_body(route: respx.Route) -> dict[str, object]:
    return _BODY.validate_json(route.calls.last.request.content)


@pytest.mark.asyncio
async def test_completion_with_vector_store_ids_prepends_the_kb_context_block(respx_mock: respx.MockRouter) -> None:
    _kb_route(respx_mock)
    anthropic: Final = respx_mock.post(_ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=_ANTHROPIC_MESSAGE))

    await litellm.acompletion(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "what is litellm?"}],
        vector_store_ids=[_KB_ID],
    )

    messages: Final = TypeAdapter(tuple[dict[str, object], ...]).validate_python(_sent_body(anthropic)["messages"])
    content: Final = TypeAdapter(tuple[dict[str, str], ...]).validate_python(messages[0]["content"])
    assert anthropic.call_count == 1
    assert [block["type"] for block in content] == ["text", "text"]
    assert content[0]["text"] == f"{_PREFIX}{_KB_TEXT}\n\n"
    assert content[1]["text"] == "what is litellm?"


@pytest.mark.asyncio
async def test_streaming_completion_carries_the_search_results_on_a_chunk_delta(respx_mock: respx.MockRouter) -> None:
    _kb_route(respx_mock)
    respx_mock.post(_ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, text=_ANTHROPIC_STREAM, headers={"content-type": "text/event-stream"})
    )

    response: Final = await litellm.acompletion(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "what is litellm?"}],
        vector_store_ids=[_KB_ID],
        stream=True,
    )
    chunks: Final = tuple([chunk async for chunk in response])
    choices: Final = tuple(itertools.chain.from_iterable(chunk.choices for chunk in chunks))
    annotated: Final = tuple(
        choice.delta.provider_specific_fields["search_results"]
        for choice in choices
        if choice.delta.provider_specific_fields and "search_results" in choice.delta.provider_specific_fields
    )

    assert len(chunks) > 0
    assert len(annotated) >= 1
    assert annotated[0][0]["object"] == "vector_store.search_results.page"
    assert annotated[0][0]["data"][0]["content"][0]["text"] == _KB_TEXT


@pytest.mark.asyncio
async def test_file_search_filters_reach_the_kb_as_a_bedrock_equals_filter(respx_mock: respx.MockRouter) -> None:
    kb: Final = _kb_route(respx_mock)
    respx_mock.post(_ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=_ANTHROPIC_MESSAGE))

    response: Final = await litellm.acompletion(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "what is litellm?"}],
        max_tokens=10,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [_KB_ID],
                "filters": {"key": "user_id", "value": "fake-user-id", "operator": "eq"},
            }
        ],
    )

    retrieval: Final = TypeAdapter(dict[str, dict[str, dict[str, object]]]).validate_python(
        _sent_body(kb)["retrievalConfiguration"]
    )
    assert retrieval["vectorSearchConfiguration"]["filter"] == {"equals": {"key": "user_id", "value": "fake-user-id"}}
    assert response.choices[0].message.content == "LiteLLM simplifies LLM access."


def _openai_messages(route: respx.Route) -> tuple[_ChatMessage, ...]:
    return TypeAdapter(tuple[_ChatMessage, ...]).validate_python(_sent_body(route)["messages"])


@pytest.mark.asyncio
async def test_openai_request_with_vector_store_ids_leads_with_a_kb_context_user_message(
    respx_mock: respx.MockRouter,
) -> None:
    _kb_route(respx_mock)
    openai: Final = respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(200, json=_OPENAI_COMPLETION))

    await litellm.acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "what is litellm?"}],
        vector_store_ids=[_KB_ID],
    )

    assert _openai_messages(openai) == (
        _ChatMessage(role="user", content=f"{_PREFIX}{_KB_TEXT}\n\n"),
        _ChatMessage(role="user", content="what is litellm?"),
    )


@pytest.mark.asyncio
async def test_a_managed_file_search_tool_is_resolved_locally_and_not_sent_upstream(
    respx_mock: respx.MockRouter,
) -> None:
    _kb_route(respx_mock)
    openai: Final = respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(200, json=_OPENAI_COMPLETION))

    await litellm.acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "what is litellm?"}],
        tools=[{"type": "file_search", "vector_store_ids": [_KB_ID]}],
    )

    assert _openai_messages(openai)[0] == _ChatMessage(role="user", content=f"{_PREFIX}{_KB_TEXT}\n\n")
    assert "tools" not in _sent_body(openai)


@pytest.mark.asyncio
async def test_an_unknown_vector_store_tool_is_forwarded_while_the_known_one_is_resolved(
    respx_mock: respx.MockRouter,
) -> None:
    _kb_route(respx_mock)
    openai: Final = respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(200, json=_OPENAI_COMPLETION))

    await litellm.acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "what is litellm?"}],
        tools=[
            {"type": "file_search", "vector_store_ids": [_KB_ID]},
            {"type": "file_search", "vector_store_ids": ["unknownVS"]},
        ],
    )

    assert _openai_messages(openai)[0] == _ChatMessage(role="user", content=f"{_PREFIX}{_KB_TEXT}\n\n")
    assert _sent_body(openai)["tools"] == [{"type": "file_search", "vector_store_ids": ["unknownVS"]}]


def _authorized_key() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-kb-proxy", user_id="kb-proxy-user")


class _SearchResultsPage(TypedDict):
    object: ReadOnly[str]
    search_query: ReadOnly[str]
    data: ReadOnly[list[dict[str, object]]]


class _ProviderFields(TypedDict):
    search_results: ReadOnly[list[_SearchResultsPage]]


class _ProxyMessage(TypedDict):
    role: ReadOnly[str]
    content: ReadOnly[str]
    provider_specific_fields: ReadOnly[_ProviderFields]


class _ProxyChoice(TypedDict):
    message: ReadOnly[_ProxyMessage]


class _ProxyCompletion(TypedDict):
    choices: ReadOnly[list[_ProxyChoice]]


@pytest.mark.asyncio
async def test_proxy_http_response_keeps_the_kb_search_results_in_provider_specific_fields(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    kb: Final = _kb_route(respx_mock)
    upstream: Final = respx_mock.post(_OPENAI_URL).mock(return_value=httpx.Response(200, json=_OPENAI_COMPLETION))
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {"model_name": "gpt-5-mini", "litellm_params": {"model": "openai/gpt-5-mini", "api_key": "sk-fixture"}}
            ],
            num_retries=0,
        ),
    )
    monkeypatch.setitem(proxy_server.app.dependency_overrides, user_api_key_auth, _authorized_key)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(proxy_server.app), base_url="http://kb-proxy.test"
    ) as client:
        result: Final = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "what is litellm?"}],
                "vector_store_ids": [_KB_ID],
            },
        )

    assert result.status_code == 200, result.text
    assert kb.call_count == 1
    assert upstream.call_count == 1
    message: Final = TypeAdapter(_ProxyCompletion).validate_json(result.content)["choices"][0]["message"]
    assert message["content"] == "ok"
    pages: Final = message["provider_specific_fields"]["search_results"]
    assert [page["object"] for page in pages] == ["vector_store.search_results.page"]
    assert pages[0]["search_query"] == "what is litellm?"
    assert pages[0]["data"]
    assert _KB_TEXT in json.dumps(pages[0]["data"])
