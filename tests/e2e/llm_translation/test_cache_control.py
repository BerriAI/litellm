"""Live e2e: provider-specific /chat/completions features take real effect.

Each case asserts the feature actually happened, not just a 200. Coverage matrix
(register-on-demand deployments, deleted on teardown):

- Bedrock (anthropic claude-haiku-4-5): prompt caching. A large cacheable prefix
  marked with ``cache_control`` is sent twice; the second call must report
  cache-read usage tokens > 0. service_tier is out of scope for Bedrock; AWS
  Bedrock does not expose an OpenAI-style request service tier, so that cell is
  intentionally not covered here.
- Vertex (gemini-2.5-flash): explicit context caching via ``cache_control``
  with a 5-minute ttl. litellm builds the Vertex cache before the generate
  call, so a never-seen prefix must come back cached on its very first call
  (Gemini's implicit caching cannot hit a cold prefix), the cached count must
  cover the marked block, and the spend row must be billed below the uncached
  price of the prompt.
- Anthropic (claude-haiku-4-5, direct): the same ``cache_control`` prefix over
  the OpenAI-compatible route; the second call must report cache-read tokens > 0.
- OpenAI (gpt-5.6): automatic prompt caching needs no request marker, so the
  cacheable prefix goes out as a plain system string with a ``prompt_cache_key``
  and the second call must report ``prompt_tokens_details.cached_tokens`` > 0.

service_tier lives in test_provider_features_e2e.py.

The provider-native cache_control request shape is not expressible with the
shared ``ChatBody`` (whose content is a plain string), so the cacheable body is
built from the typed content blocks shared in ``endpoints_client.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import Result, UnknownApiError, unwrap
from endpoints_client import CacheControl, RichMessage, TextBlock
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody, Usage
from passthrough_client import PassthroughClient
import os

pytestmark = pytest.mark.e2e

BEDROCK_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
VERTEX_MODEL = "vertex_ai/gemini-2.5-flash"
ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5-20251001"
OPENAI_MODEL = "openai/gpt-5.6"
VERTEX_CACHE_TTL: Final = "300s"
VERTEX_COLD_CALL_ATTEMPTS: Final = 3
VERTEX_MINIMUM_CACHED_TOKENS: Final = 1024
CACHED_SHARE_OF_PROMPT: Final = 0.9
VERTEX_CACHE_REJECTION_MARKER: Final = "minimum token count to start explicit caching"


class CacheChatBody(BaseModel):
    model: str
    messages: list[RichMessage]
    max_tokens: int = 64
    cache: dict[str, bool] = {"no-cache": True}


def _cacheable_prefix() -> str:
    """A prefix long enough to clear provider minimum cacheable sizes (Haiku is
    2048 tokens), unique per run so the first call writes and the second reads."""
    marker = unique_marker()
    body = " ".join(
        f"Cacheable reference paragraph {index} for run {marker}." for index in range(600)
    )
    return f"{body}\nEnd of reference material {marker}."


def _cached_read_tokens(usage: Usage | None) -> int:
    """Cache-read tokens however the provider reports them: Anthropic-style
    ``cache_read_input_tokens`` or OpenAI-style ``prompt_tokens_details.cached_tokens``."""
    if usage is None:
        return 0
    if usage.cache_read_input_tokens:
        return usage.cache_read_input_tokens
    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
        return usage.prompt_tokens_details.cached_tokens
    return 0


def _cache_chat(
    client: PassthroughClient, key: str, model: str, prefix: str, ttl: str | None = None
) -> Result[ChatResponse]:
    body = CacheChatBody(
        model=model,
        messages=[
            RichMessage(
                role="system",
                content=[TextBlock(text=prefix, cache_control=CacheControl(ttl=ttl))],
            ),
            RichMessage(role="user", content=[TextBlock(text="Reply with one word.")]),
        ],
    )
    return client.proxy.transport.post(
        "/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=body,
        response_type=ChatResponse,
    )


def _plain_cache_chat(
    client: PassthroughClient, key: str, model: str, prefix: str, cache_key: str
) -> Result[ChatResponse]:
    """The same cacheable prefix as a plain system string, for providers that cache
    automatically and take no per-block marker (OpenAI)."""
    return client.proxy.chat(
        key,
        ChatBody(
            model=model,
            messages=[
                ChatMessage(role="system", content=prefix),
                ChatMessage(role="user", content="Reply with one word."),
            ],
            max_tokens=64,
            prompt_cache_key=cache_key,
        ),
    )


def _assert_cache_read_on_second_call(
    model: str, send: Callable[[str], Result[ChatResponse]]
) -> None:
    prefix = _cacheable_prefix()

    first = unwrap(send(prefix))
    assert first.choices, f"{model}: first cache-priming call returned no choices: {first}"

    deadline = time.monotonic() + 30.0
    while True:
        second = unwrap(send(prefix))
        read_tokens = _cached_read_tokens(second.usage)
        if read_tokens > 0 or time.monotonic() >= deadline:
            break
        time.sleep(3.0)

    assert read_tokens > 0, (
        f"{model}: second identical call reported no cache-read tokens "
        f"({second.usage}); prompt caching did not take effect"
    )


def _cold_cache_call(send: Callable[[str], Result[ChatResponse]]) -> ChatResponse | None:
    result: Final = send(_cacheable_prefix())
    match result:
        case UnknownApiError(status_code=400, body=body) if VERTEX_CACHE_REJECTION_MARKER in body:
            return None
        case _:
            return unwrap(result)


def _first_cold_call_reads_cache(model: str, send: Callable[[str], Result[ChatResponse]]) -> ChatResponse:
    completion: Final = next(
        (
            candidate
            for candidate in (_cold_cache_call(send) for _ in range(VERTEX_COLD_CALL_ATTEMPTS))
            if candidate is not None and _cached_read_tokens(candidate.usage) >= VERTEX_MINIMUM_CACHED_TOKENS
        ),
        None,
    )
    assert completion is not None, (
        f"{model}: {VERTEX_COLD_CALL_ATTEMPTS} never-seen prompts marked with cache_control were each either "
        f"rejected by Vertex's minimum-token check or served with fewer than {VERTEX_MINIMUM_CACHED_TOKENS} "
        "cached tokens on their first call; explicit context caching did not engage"
    )
    assert completion.choices, f"{model}: cached call returned no choices: {completion}"
    usage: Final = completion.usage
    cached: Final = _cached_read_tokens(usage)
    assert usage and usage.prompt_tokens and cached >= CACHED_SHARE_OF_PROMPT * usage.prompt_tokens, (
        f"{model}: only {cached} of {usage.prompt_tokens if usage else None} prompt tokens were served from the "
        "cache; the cache_control block was not cached whole"
    )
    return completion


def _input_rate(client: PassthroughClient, model: str) -> float:
    entry: Final = next((row for row in client.proxy.model_info() if row.model_name == model), None)
    assert entry and entry.model_info.input_cost_per_token, f"/model/info resolved no input rate for {model}"
    return entry.model_info.input_cost_per_token


def _assert_billed_below_uncached_prompt(client: PassthroughClient, model: str, completion: ChatResponse) -> None:
    assert completion.id, f"{model}: cached completion carried no id to find its spend row by"
    usage: Final = completion.usage
    assert usage and usage.prompt_tokens, f"{model}: cached completion carried no prompt_tokens: {usage}"
    rows: Final = client.proxy.poll_logs_for_request_id(completion.id, predicate=lambda rs: (rs[0].spend or 0) > 0)
    assert rows, f"{model}: no costed /spend/logs row for request {completion.id}"
    row: Final = rows[0]
    assert row.prompt_tokens == usage.prompt_tokens, (
        f"{model}: spend row prompt_tokens {row.prompt_tokens} != response prompt_tokens {usage.prompt_tokens}"
    )
    uncached_prompt_cost: Final = usage.prompt_tokens * _input_rate(client, model)
    assert row.spend is not None and row.spend < uncached_prompt_cost, (
        f"{model}: spend {row.spend} is not below the uncached price of the prompt alone ({uncached_prompt_cost} for "
        f"{usage.prompt_tokens} tokens); cache-read pricing was not applied"
    )


class TestCacheControl:
    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.prompt_cache_5m.nonstream.works",
        exercised_on=[],
    )
    def test_bedrock_prompt_caching_reads_cache(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-bedrock-cache-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=BEDROCK_MODEL, aws_region_name="us-east-1"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        _assert_cache_read_on_second_call(model, lambda prefix: _cache_chat(client, key, model, prefix))

    @pytest.mark.covers(
        "llm.chat_completions.vertex.prompt_cache_5m.nonstream.works",
        exercised_on=[],
    )
    def test_vertex_prompt_caching_reads_cache(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-vertex-cache-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=VERTEX_MODEL,
                vertex_project=os.environ.get("VERTEXAI_PROJECT"),
                vertex_location="us-central1",
                vertex_credentials=os.environ.get("VERTEXAI_CREDENTIALS"),
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        completion: Final = _first_cold_call_reads_cache(
            model, lambda prefix: _cache_chat(client, key, model, prefix, ttl=VERTEX_CACHE_TTL)
        )
        _assert_billed_below_uncached_prompt(client, model, completion)

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.prompt_cache_5m.nonstream.works",
        exercised_on=[],
    )
    def test_anthropic_prompt_caching_reads_cache(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-anthropic-cache-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=ANTHROPIC_MODEL, api_key="os.environ/ANTHROPIC_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        _assert_cache_read_on_second_call(model, lambda prefix: _cache_chat(client, key, model, prefix))

    @pytest.mark.covers(
        "llm.chat_completions.openai.prompt_cache_5m.nonstream.works",
        exercised_on=[],
    )
    def test_openai_prompt_caching_reads_cache(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-openai-cache-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=OPENAI_MODEL, api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        cache_key = f"e2e-openai-cache-{unique_marker()}"
        _assert_cache_read_on_second_call(
            model, lambda prefix: _plain_cache_chat(client, key, model, prefix, cache_key)
        )
