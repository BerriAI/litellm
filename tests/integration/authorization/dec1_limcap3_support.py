"""Shared pieces for the ``test_dec1_limcap3_*`` contracts (rate limits and model capacity, round 3).

Round 3 works from the "Not covered" and "Suspected bugs" lists of the ``test_dec1_ratelim_``,
``test_dec1_limcap2_``, ``test_dec1_budlim2_`` and capacity families. The provider, probes, refusal assertions and
Redis readers come from ``dec1_limits_support`` and ``dec1_ratelim_support``, unchanged.

What is added here: a provider that also answers Cohere ``/rerank`` (with ``meta.tokens`` and
``meta.billed_units``), a provider whose chat usage carries only ``total_tokens``, a Google GenAI native
``generateContent`` request helper, and small readers for the key token window.

Every literal is read from the Python product on main (49b2058c): the v3 limiter
(``litellm/proxy/hooks/parallel_request_limiter_v3.py``, Rust port in draft PR #51537) and the dynamic rate
limiter v3 (``litellm/proxy/hooks/dynamic_rate_limiter_v3.py``, Rust port in draft PR #51540).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually
from tests.integration._support.wire import Reply, Request, Wire
from tests.integration.authorization.dec1_limits_support import (
    DEFAULT_USAGE,
    Usage,
    counter_key,
    hashed,
    probe,
    provider,
    redis_integer,
    shared_redis,
)

RERANK_MODEL: Final = "cohere/rerank-v3.5"
RERANK_INPUT_TOKENS: Final = 30
RERANK_BILLED_TOTAL: Final = 24
TOTAL_ONLY_TOKENS: Final = 25
COMPLETION_MODEL: Final = "text-completion-openai/gpt-3.5-turbo-instruct"


def unique(label: str) -> str:
    return f"limcap3-{label}-{uuid.uuid4().hex[:10]}"


def rerank_reply(*, meta: Mapping[str, JsonValue]) -> Reply:
    """A Cohere v2 rerank answer with the given ``meta`` block."""
    return Reply(
        body=json.dumps(
            {
                "id": f"rerank-{uuid.uuid4().hex[:8]}",
                "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}],
                "meta": {"api_version": {"version": "2"}, **dict(meta)},
            }
        ).encode()
    )


def rerank_provider(meta: Mapping[str, JsonValue] | None = None) -> Callable[[Request], Reply]:
    """The limits provider, plus ``POST .../rerank`` answered with ``meta`` (by default
    ``tokens.input_tokens = 30``, the field ``_resolve_rerank_token_usage`` reads first)."""
    base: Final = provider()
    chosen: Final[Mapping[str, JsonValue]] = (
        meta
        if meta is not None
        else {"tokens": {"input_tokens": RERANK_INPUT_TOKENS, "output_tokens": 0}, "billed_units": {"search_units": 1}}
    )

    def respond(request: Request) -> Reply:
        if request.method == "POST" and request.target.split("?", 1)[0].endswith("/rerank"):
            return rerank_reply(meta=chosen)
        return base(request)

    return respond


def total_only_provider() -> Callable[[Request], Reply]:
    """A chat provider whose usage reports ``prompt_tokens: 0``, ``completion_tokens: 0`` and
    ``total_tokens: 25``, the shape a source that cannot split its usage reports."""

    def respond(request: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-limcap3",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "limcap3"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": TOTAL_ONLY_TOKENS},
                }
            ).encode()
        )

    return respond


def cached_provider(usage: Usage) -> Callable[[Request], Reply]:
    """The limits provider reporting ``usage`` (with ``prompt_tokens_details.cached_tokens`` when set)."""
    return provider(usage)


def rerank_model(scenario: Scenario, wire: Wire, **extra: JsonValue) -> str:
    """A Cohere rerank deployment that calls the wire."""
    return scenario.model(model=RERANK_MODEL, api_base=wire.url, api_key="synthetic-cohere-key", **extra)


def completion_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model=COMPLETION_MODEL, api_base=f"{wire.url}/v1")


def rerank(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/rerank",
        {"model": model, "query": probe(), "documents": ["first limcap3 document", "second limcap3 document"]},
        key=key,
    )


def generate_content(
    gateway: Gateway, model: str, key: str, generation_config: Mapping[str, JsonValue] | None = None
) -> httpx.Response:
    """``POST /v1beta/models/<model>:generateContent`` with one 40 character user part (10 estimated tokens)."""
    body: Final[dict[str, JsonValue]] = {
        "contents": [{"role": "user", "parts": [{"text": probe()}]}],
        **({"generationConfig": dict(generation_config)} if generation_config is not None else {}),
    }
    return gateway.request("POST", f"/v1beta/models/{model}:generateContent", body, key=key)


def key_tokens(key: str) -> int | None:
    with shared_redis() as redis:
        return redis_integer(redis, counter_key("api_key", hashed(key), "tokens"))


def key_tokens_settle_to(key: str, expected: int, seconds: float = 20) -> int | None:
    """Poll ``{api_key:<hash>}:tokens`` until it holds ``expected``; return what it holds."""
    with shared_redis() as redis:
        name: Final = counter_key("api_key", hashed(key), "tokens")
        return eventually(
            lambda: redis_integer(redis, name),
            lambda value: value == expected,
            seconds=seconds,
            return_last_on_timeout=True,
        )


def key_requests(key: str) -> int | None:
    with shared_redis() as redis:
        return redis_integer(redis, counter_key("api_key", hashed(key), "requests"))


__all__ = (
    "COMPLETION_MODEL",
    "DEFAULT_USAGE",
    "RERANK_BILLED_TOTAL",
    "RERANK_INPUT_TOKENS",
    "RERANK_MODEL",
    "TOTAL_ONLY_TOKENS",
    "cached_provider",
    "completion_model",
    "generate_content",
    "key_requests",
    "key_tokens",
    "key_tokens_settle_to",
    "rerank",
    "rerank_model",
    "rerank_provider",
    "rerank_reply",
    "total_only_provider",
    "unique",
)
