import itertools
import json
from collections.abc import Callable, Iterator, Mapping
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME

PERPLEXITY_SEARCH_URL: Final = "https://api.perplexity.ai/search"
ANTHROPIC_MESSAGES_URL: Final = "https://api.anthropic.com/v1/messages"
OPENAI_CHAT_URL: Final = "https://api.openai.com/v1/chat/completions"
OPENAI_RESPONSES_URL: Final = "https://api.openai.com/v1/responses"
FINAL_ANSWER: Final = "Here is what the searches found."
RUNAWAY_LOOP_LIMIT: Final = 8
SEARCH_LIMIT_ERROR_CODE: Final = "max_uses_exceeded"


@pytest.fixture
def perplexity_search(monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter) -> respx.Route:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "test-perplexity-key")
    monkeypatch.delenv("PERPLEXITY_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    return respx_mock.post(PERPLEXITY_SEARCH_URL).respond(
        json={"results": [{"title": "AI news", "url": "https://example.com/ai-news", "snippet": "AI news."}]}
    )


def request_bodies(route: respx.Route) -> list[dict]:
    return [json.loads(call.request.content) for call in route.calls]


def offered_tool_names(body: Mapping[str, object]) -> list[str]:
    tools = body.get("tools") or []
    return [tool.get("function", {}).get("name") or tool.get("name") for tool in tools]


def _searching_model(
    reply: Callable[[dict, int, bool, str], dict],
) -> Callable[[httpx.Request], httpx.Response]:
    """
    Searches while the search tool is offered. Once it is not, answers only if the
    history says the search limit was reached, and otherwise ends the turn with no
    text, which is what Claude did live when it still wanted to search.
    """
    counter: Final[Iterator[int]] = itertools.count(1)

    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        call_number: Final = next(counter)
        searches: Final = LITELLM_WEB_SEARCH_TOOL_NAME in offered_tool_names(body) and call_number < RUNAWAY_LOOP_LIMIT
        answer: Final = FINAL_ANSWER if SEARCH_LIMIT_ERROR_CODE in request.content.decode() else ""
        return httpx.Response(200, json=reply(body, call_number, searches, answer))

    return respond


def _anthropic_reply(body: dict, call_number: int, searches: bool, answer: str) -> dict:
    content: Final = (
        [
            {
                "type": "tool_use",
                "id": f"toolu_{call_number}",
                "name": LITELLM_WEB_SEARCH_TOOL_NAME,
                "input": {"query": f"latest ai news, take {call_number}"},
            }
        ]
        if searches
        else [{"type": "text", "text": answer}]
    )
    return {
        "id": f"msg_{call_number}",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": content,
        "stop_reason": "tool_use" if searches else "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _openai_chat_reply(body: dict, call_number: int, searches: bool, answer: str) -> dict:
    message: Final = (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{call_number}",
                    "type": "function",
                    "function": {
                        "name": LITELLM_WEB_SEARCH_TOOL_NAME,
                        "arguments": json.dumps({"query": f"latest ai news, take {call_number}"}),
                    },
                }
            ],
        }
        if searches
        else {"role": "assistant", "content": answer}
    )
    return {
        "id": f"chatcmpl-{call_number}",
        "object": "chat.completion",
        "created": 1767225600,
        "model": body["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if searches else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _openai_responses_reply(body: dict, call_number: int, searches: bool, answer: str) -> dict:
    output: Final = (
        [
            {
                "type": "function_call",
                "id": f"fc_{call_number}",
                "call_id": f"call_{call_number}",
                "name": LITELLM_WEB_SEARCH_TOOL_NAME,
                "arguments": json.dumps({"query": f"latest ai news, take {call_number}"}),
                "status": "completed",
            }
        ]
        if searches
        else [
            {
                "type": "message",
                "id": f"msg_{call_number}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer, "annotations": []}],
            }
        ]
    )
    return {
        "id": f"resp_{call_number}",
        "object": "response",
        "created_at": 1767225600,
        "status": "completed",
        "model": body["model"],
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def fresh_http_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture
def anthropic_model(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, fresh_http_clients: None
) -> respx.Route:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    return respx_mock.post(ANTHROPIC_MESSAGES_URL).mock(side_effect=_searching_model(_anthropic_reply))


@pytest.fixture
def openai_chat_model(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, fresh_http_clients: None
) -> respx.Route:
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return respx_mock.post(OPENAI_CHAT_URL).mock(side_effect=_searching_model(_openai_chat_reply))


@pytest.fixture
def openai_responses_model(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, fresh_http_clients: None
) -> respx.Route:
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return respx_mock.post(OPENAI_RESPONSES_URL).mock(side_effect=_searching_model(_openai_responses_reply))
