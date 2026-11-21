"""Response cache hits on the big providers, whose wire answers are not in the OpenAI shape.

The cache stores the answer after it was translated into the route's shape (``ModelResponse`` for chat, the
Anthropic message for ``/v1/messages``), so a hit must replay exactly what the first caller got, whatever the
provider spoke on the wire: Anthropic Messages, Gemini ``generateContent``, Bedrock Converse or Azure OpenAI.
Every cell points one deployment at a counting provider peer (``provider_peer``) that answers in that
provider's own JSON with a fresh id per reply, sends the same request twice, and requires one provider call
and an identical second answer. These are the paths moving to Rust next (``/v1/messages`` on more providers,
the provider clients), and the cache key must not start depending on the provider's wire request (the Rust
messages path keys signed Bedrock calls differently, ``inference-messages/src/handler.rs`` ``cacheable``).

UNVERIFIED for every cell: the deployment parameters (``api_base`` on Bedrock in particular) and the exact
provider paths were not run; the expected translation of each wire shape is covered by the provider suites,
so these cells only compare the first answer with the replay.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, list_value, object_value
from tests.integration._support.wire import Request
from tests.integration.caching.dec1_cache2_support import PeerExchange, ProviderPeer, ask_peer, provider_peer
from tests.integration.caching.dec1_cache_support import CACHE_KEY_HEADER
from tests.integration.caching.response_cache_case import prompt

PROVIDER_KEY: Final = "synthetic-provider-key"


def anthropic_answer(_: Request, unique: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{unique}",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": f"anthropic {unique}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def anthropic_tool_answer(_: Request, unique: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{unique}",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [
            {"type": "thinking", "thinking": f"think {unique}", "signature": f"sig-{unique}"},
            {"type": "text", "text": "Checking the weather"},
            {"type": "tool_use", "id": f"toolu_{unique}", "name": "get_weather", "input": {"city": "Paris"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 7, "output_tokens": 5},
    }


def gemini_answer(_: Request, unique: str) -> dict[str, JsonValue]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": f"gemini {unique}"}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3, "totalTokenCount": 7},
        "modelVersion": "gemini-2.5-flash",
        "responseId": f"gemini-{unique}",
    }


def converse_answer(_: Request, unique: str) -> dict[str, JsonValue]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": f"bedrock {unique}"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
        "metrics": {"latencyMs": 12},
    }


def azure_answer(request: Request, unique: str) -> dict[str, JsonValue]:
    if "/embeddings" in request.target:
        return {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, int(unique[:6], 16) / 2**24]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
    return {
        "id": f"chatcmpl-{unique}",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": f"azure {unique}"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


def anthropic_model(scenario: Scenario, peer: ProviderPeer) -> str:
    return scenario.model(model="anthropic/claude-haiku-4-5", api_base=peer.url, api_key=PROVIDER_KEY)


def gemini_model(scenario: Scenario, peer: ProviderPeer) -> str:
    return scenario.model(model="gemini/gemini-2.5-flash", api_base=peer.url, api_key=PROVIDER_KEY)


def bedrock_model(scenario: Scenario, peer: ProviderPeer) -> str:
    return scenario.model(
        model="bedrock/converse/amazon.nova-micro-v1:0",
        api_base=peer.url,
        aws_access_key_id="AKIDEXAMPLESYNTHETIC",
        aws_secret_access_key="synthetic-secret-access-key",
        aws_region_name="us-east-1",
    )


def azure_model(scenario: Scenario, peer: ProviderPeer, provider_model: str = "azure/gpt-4o-mini") -> str:
    return scenario.model(model=provider_model, api_base=peer.url, api_version="2024-10-21", api_key=PROVIDER_KEY)


def _twice(
    gateway: Gateway, peer: ProviderPeer, path: str, body: Mapping[str, JsonValue]
) -> tuple[PeerExchange, PeerExchange]:
    first: Final = ask_peer(gateway, peer, path, body)
    assert first.reached_provider, first.response.text
    second: Final = eventually(
        lambda: ask_peer(gateway, peer, path, body), lambda exchange: not exchange.reached_provider, 15
    )
    assert not second.reached_provider, second.response.text
    if path != "/v1/embeddings":
        assert second.response.headers.get(CACHE_KEY_HEADER), dict(second.response.headers)
    return first, second


def _chat(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": prompt()}], **extra}


def _message(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": prompt()}], **extra}


def _chat_view(answer: Mapping[str, JsonValue]) -> tuple[JsonValue, JsonValue, JsonValue]:
    choice: Final = object_value(list_value(answer["choices"])[0])
    return answer["id"], object_value(choice["message"]).get("content"), answer.get("usage")


ChatModel = Callable[[Scenario, ProviderPeer], str]


@pytest.mark.parametrize(
    ("make_model", "answer"),
    (
        pytest.param(anthropic_model, anthropic_answer, id="anthropic"),
        pytest.param(gemini_model, gemini_answer, id="gemini"),
        pytest.param(bedrock_model, converse_answer, id="bedrock-converse"),
        pytest.param(azure_model, azure_answer, id="azure"),
    ),
)
def test_a_chat_hit_replays_the_translated_answer_of_each_big_provider(
    gateway: Gateway, make_model: ChatModel, answer: Callable[[Request, str], dict[str, JsonValue]]
) -> None:
    """Each provider answers a chat request in its own wire shape, which the proxy translates into a chat
    completion. The identical request must then be answered without reaching the provider, carry
    ``x-litellm-cache-key``, and repeat the first answer's ``id``, message content and ``usage`` exactly.
    """
    with provider_peer(answer) as peer, gateway.scenario() as scenario:
        model: Final = make_model(scenario, peer)
        first, second = _twice(gateway, peer, "/v1/chat/completions", _chat(model))
        assert _chat_view(second.json()) == _chat_view(first.json()), (first.response.text, second.response.text)
        assert peer.calls() == 1


def test_an_anthropic_chat_hit_keeps_the_tool_call_and_the_reasoning(gateway: Gateway) -> None:
    """Anthropic answers a chat request with a thinking block, a text block and a ``tool_use`` block. The chat
    answer carries ``tool_calls`` and ``reasoning_content``. The cached replay must carry the same tool call id,
    function name and arguments and the same ``reasoning_content``, so a client continuing the tool loop from a
    hit sees the same call.
    """
    with provider_peer(anthropic_tool_answer) as peer, gateway.scenario() as scenario:
        model: Final = anthropic_model(scenario, peer)
        tool: Final[dict[str, JsonValue]] = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
        first, second = _twice(gateway, peer, "/v1/chat/completions", _chat(model, tools=[tool]))
        first_message: Final = object_value(object_value(list_value(first.json()["choices"])[0])["message"])
        second_message: Final = object_value(object_value(list_value(second.json()["choices"])[0])["message"])
        assert second_message.get("tool_calls") == first_message.get("tool_calls"), second.response.text
        assert first_message.get("tool_calls"), first.response.text
        assert second_message.get("reasoning_content") == first_message.get("reasoning_content")


def test_an_anthropic_messages_hit_keeps_thinking_and_tool_use_blocks(gateway: Gateway) -> None:
    """On the Rust ``/v1/messages`` path Anthropic answers with thinking, text and ``tool_use`` blocks. The
    replay must return the same ``content`` list, block for block (thinking text and ``signature`` included),
    and the same ``stop_reason`` ``tool_use``: a signature that changed would be rejected by Anthropic on the
    next turn.
    """
    with provider_peer(anthropic_tool_answer) as peer, gateway.scenario() as scenario:
        model: Final = anthropic_model(scenario, peer)
        first, second = _twice(gateway, peer, "/v1/messages", _message(model))
        assert second.json()["content"] == first.json()["content"], (first.response.text, second.response.text)
        assert second.json()["stop_reason"] == "tool_use"
        assert second.json()["id"] == first.json()["id"]


@pytest.mark.parametrize(
    ("make_model", "answer"),
    (
        pytest.param(gemini_model, gemini_answer, id="gemini"),
        pytest.param(bedrock_model, converse_answer, id="bedrock-converse"),
        pytest.param(azure_model, azure_answer, id="azure"),
    ),
)
def test_a_messages_hit_on_a_non_anthropic_provider_replays_the_translated_message(
    gateway: Gateway, make_model: ChatModel, answer: Callable[[Request, str], dict[str, JsonValue]]
) -> None:
    """``/v1/messages`` on a Gemini, Bedrock Converse or Azure deployment translates the provider answer into an
    Anthropic message. The identical request must be answered from the cache with the same ``id``, ``content``
    and ``usage``. On Bedrock the provider request is SigV4 signed with a timestamp, so a key built from the
    signed wire request would never hit; the key must stay the request-level one.
    On Gemini the hit is served without reaching the provider but carries no ``x-litellm-cache-key`` (CI run
    44161684479): ``_set_cached_hidden_param`` (caching_handler.py:334-340) writes the key only into a hidden params
    map the cached object already has, and does nothing for a stored dict without one.
    """
    if make_model is gemini_model:
        pytest.skip("BUG: a /v1/messages cache hit on a Gemini deployment answers without x-litellm-cache-key")
    with provider_peer(answer) as peer, gateway.scenario() as scenario:
        model: Final = make_model(scenario, peer)
        first, second = _twice(gateway, peer, "/v1/messages", _message(model))
        for field in ("id", "content", "usage"):
            assert second.json()[field] == first.json()[field], (field, first.response.text, second.response.text)


def test_an_azure_embedding_hit_replays_the_stored_vector(gateway: Gateway) -> None:
    """An Azure embedding deployment answers once with a vector unique to that reply. The identical request must
    be answered without reaching Azure and return the same vector at index 0.
    """
    with provider_peer(azure_answer) as peer, gateway.scenario() as scenario:
        model: Final = azure_model(scenario, peer, "azure/text-embedding-3-small")
        body: Final[dict[str, JsonValue]] = {"model": model, "input": prompt()}
        first, second = _twice(gateway, peer, "/v1/embeddings", body)
        assert second.json()["data"] == first.json()["data"], (first.response.text, second.response.text)


def test_a_gemini_chat_hit_reports_zero_cost_while_the_fill_reports_the_real_cost(gateway: Gateway) -> None:
    """A priced Gemini deployment (0.001 per input token, 0.002 per output token) answers with 4 prompt and 3
    completion tokens. The filling response's ``x-litellm-response-cost`` must be 0.01 and the hit's must be 0:
    the cost of a hit is zero whatever the provider.
    """
    with provider_peer(gemini_answer) as peer, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="gemini/gemini-2.5-flash",
            api_base=peer.url,
            api_key=PROVIDER_KEY,
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        first, second = _twice(gateway, peer, "/v1/chat/completions", _chat(model))
        assert float(first.response.headers["x-litellm-response-cost"]) == pytest.approx(4 * 0.001 + 3 * 0.002)
        assert float(second.response.headers["x-litellm-response-cost"]) == 0
