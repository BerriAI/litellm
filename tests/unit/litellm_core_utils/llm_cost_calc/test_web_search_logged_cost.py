import asyncio
import json
from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger

_MODEL: Final = "gpt-sized-search-unit"
_INPUT_COST: Final = 1e-06
_OUTPUT_COST: Final = 4e-06
_PER_QUERY: Final = {
    "search_context_size_low": 0.011,
    "search_context_size_medium": 0.022,
    "search_context_size_high": 0.033,
}
_PROMPT_TOKENS: Final = 100
_COMPLETION_TOKENS: Final = 20
_USAGE: Final = {
    "prompt_tokens": _PROMPT_TOKENS,
    "completion_tokens": _COMPLETION_TOKENS,
    "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_CHAT_RESPONSE: Final = {
    "id": "chatcmpl-search",
    "object": "chat.completion",
    "created": 1700000000,
    "model": _MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "A positive story",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "start_index": 0,
                            "end_index": 5,
                            "title": "news",
                            "url": "https://news.example/a",
                        },
                    }
                ],
            },
            "finish_reason": "stop",
        }
    ],
    "usage": _USAGE,
}
_RESPONSES_BODY: Final = {
    "id": "resp_search",
    "object": "response",
    "created_at": 1700000000,
    "status": "completed",
    "model": _MODEL,
    "output": [
        {"type": "web_search_call", "id": "ws_search", "status": "completed"},
        {
            "type": "message",
            "id": "msg_search",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "A positive story", "annotations": []}],
        },
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": _PROMPT_TOKENS,
        "output_tokens": _COMPLETION_TOKENS,
        "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
    },
}
_RESPONSES_STREAM: Final = "".join(
    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
    for event in (
        {"type": "response.created", "response": {**_RESPONSES_BODY, "status": "in_progress", "output": []}},
        {"type": "response.completed", "response": _RESPONSES_BODY},
    )
)

_ContextSize = Literal["search_context_size_low", "search_context_size_medium", "search_context_size_high"]


class _LoggedCost(TypedDict):
    response_cost: ReadOnly[float]
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]


class _CostRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: Final[list[_LoggedCost]] = []
        self.logged: Final = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.payloads.append(TypeAdapter(_LoggedCost).validate_python(kwargs["standard_logging_object"]))
        self.logged.set()


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _CostRecorder:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    entry: Final = {
        "input_cost_per_token": _INPUT_COST,
        "output_cost_per_token": _OUTPUT_COST,
        "litellm_provider": "openai",
        "mode": "chat",
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "supports_web_search": True,
        "search_context_cost_per_query": _PER_QUERY,
    }
    monkeypatch.setitem(litellm.model_cost, _MODEL, entry)
    monkeypatch.setitem(litellm.model_cost, f"openai/{_MODEL}", entry)
    cost_recorder: Final = _CostRecorder()
    monkeypatch.setattr(litellm, "callbacks", [cost_recorder])
    return cost_recorder


async def _logged_cost(recorder: _CostRecorder) -> _LoggedCost:
    await asyncio.wait_for(recorder.logged.wait(), timeout=10)
    return recorder.payloads[-1]


def _expected_cost(payload: _LoggedCost, size: _ContextSize) -> float:
    return payload["prompt_tokens"] * _INPUT_COST + payload["completion_tokens"] * _OUTPUT_COST + _PER_QUERY[size]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("web_search_options", "size"),
    [
        (None, "search_context_size_medium"),
        ({"search_context_size": "low"}, "search_context_size_low"),
        ({"search_context_size": "high"}, "search_context_size_high"),
    ],
)
async def test_chat_web_search_logged_cost_adds_the_per_query_cost_for_the_context_size(
    web_search_options: dict[str, str] | None,
    size: _ContextSize,
    recorder: _CostRecorder,
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    options: Final = {"web_search_options": web_search_options} if web_search_options is not None else {}

    await litellm.acompletion(
        model=f"openai/{_MODEL}",
        messages=[{"role": "user", "content": "What was a positive news story from today?"}],
        api_key="sk-unit-test",
        **options,
    )
    payload: Final = await _logged_cost(recorder)

    assert (payload["prompt_tokens"], payload["completion_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS)
    assert payload["response_cost"] == pytest.approx(_expected_cost(payload, size), abs=1e-12)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "size", "stream"),
    [
        ([{"type": "web_search_preview", "search_context_size": "low"}], "search_context_size_low", True),
        ([{"type": "web_search_preview", "search_context_size": "low"}], "search_context_size_low", False),
        ([{"type": "web_search_preview"}], "search_context_size_medium", True),
        ([{"type": "web_search_preview"}], "search_context_size_medium", False),
    ],
)
async def test_responses_web_search_logged_cost_adds_the_per_query_cost_for_the_context_size(
    tools: list[dict[str, str]],
    size: _ContextSize,
    stream: bool,
    recorder: _CostRecorder,
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, text=_RESPONSES_STREAM, headers={"content-type": "text/event-stream"})
        if stream
        else httpx.Response(200, json=_RESPONSES_BODY)
    )

    response: Final = await litellm.aresponses(
        model=f"openai/{_MODEL}",
        input=[{"role": "user", "content": "What was a positive news story from today?"}],
        tools=tools,
        stream=stream,
        api_key="sk-unit-test",
    )
    if stream:
        assert [event async for event in response]
    payload: Final = await _logged_cost(recorder)

    assert (payload["prompt_tokens"], payload["completion_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS)
    assert payload["response_cost"] == pytest.approx(_expected_cost(payload, size), abs=1e-12)
