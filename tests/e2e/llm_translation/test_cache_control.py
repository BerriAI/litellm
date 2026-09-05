"""Live e2e: provider-specific /chat/completions features take real effect.

Each case asserts the feature actually happened, not just a 200. Coverage matrix
(register-on-demand deployments, deleted on teardown):

- Bedrock (anthropic claude-haiku-4-5): prompt caching. A large cacheable prefix
  marked with ``cache_control`` is sent twice; the second call must report
  cache-read usage tokens > 0. service_tier is out of scope for Bedrock; AWS
  Bedrock does not expose an OpenAI-style request service tier, so that cell is
  intentionally not covered here.
- Vertex (gemini-2.5-flash): prompt caching via ``cache_control`` context
  caching; the second identical call must report cached prompt tokens > 0.
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

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import Result, unwrap
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
    client: PassthroughClient, key: str, model: str, prefix: str
) -> Result[ChatResponse]:
    body = CacheChatBody(
        model=model,
        messages=[
            RichMessage(
                role="system",
                content=[TextBlock(text=prefix, cache_control=CacheControl())],
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
        _assert_cache_read_on_second_call(model, lambda prefix: _cache_chat(client, key, model, prefix))

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
