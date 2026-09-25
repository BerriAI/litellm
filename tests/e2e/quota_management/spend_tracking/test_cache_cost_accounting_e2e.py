"""Live e2e: prompt-cache token accounting bills each cache component at its own rate.

Four regressions the gateway has shipped fixes for, pinned against real OpenAI
prompt caching (implicit, keyed on the token prefix). Every test registers its own
deployment with distinct custom rates for input / output / cache-read /
cache-creation, so the expected bill is exactly the row's token counts times the
configured rates and a component billed at the wrong rate can never pass:

- cache writes: gpt-5.6's cache-write tokens must land on the spend row as
  cache-creation tokens billed at the cache-creation rate, not silently at the
  input rate (#34046)
- breakdown components: the row's metadata.cost_breakdown must itemize cache-read,
  cache-creation, and reasoning costs, with reasoning a subset of output (#31686)
- streaming: a streamed call's reassembled usage must keep the cached-token detail
  so cache reads bill at the cache-read discount, not full input price (#34812)
- /v1/messages bridge: a request served by a Responses-only OpenAI model crosses
  the anthropic-messages -> Responses adapter and must keep its cache-read tokens
  and their discounted billing (#34957)

Each test drives the model that actually reports the component it bills, which is
not the same model throughout. gpt-5.6-luna reports cache-write tokens on every
call over the caching minimum and never reports a cache read, so it is the one
model that can prove cache-write billing and the one model that can never prove
cache-read billing. gpt-5.5 is the reverse: it reports cached tokens on the second
call and no cache writes at all. gpt-5.3-codex is Responses-only, which is what
forces the /v1/messages bridge, and it starts reporting cache reads once the
prefix is a few thousand tokens rather than one.

OpenAI caching is best-effort, so each test retries with a fresh prefix (new
marker = brand-new cache identity) up to three times before failing; the prime and
measured calls share the prefix but differ in the trailing question, which defeats
the proxy's own response cache without touching the provider's prefix cache.

The test that asserts on reasoning cost requests reasoning explicitly with
`reasoning_effort`, so that assertion rests on a parameter the test sets rather
than on whatever the model happens to do by default. Its prime call carries the
same value: OpenAI's prefix cache keys on the reasoning setting as well as the
tokens, so a prime at a different effort never produces a read.
"""

import pytest

from cost_rows import (
    CostRow,
    approx_equal,
    assert_fresh_tokens_billed_at,
    assert_total_is_sum_of_components,
    cacheable_prefix,
    poll_cost_row,
    poll_cost_row_where,
    register_priced_model,
)
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import AnthropicMessagesBody, ChatBody, ChatMessage, LiteLLMParamsBody
from pydantic import BaseModel
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

CACHE_WRITE_BACKEND = "openai/gpt-5.6-luna"
CACHE_READ_BACKEND = "openai/gpt-5.5"
BRIDGE_BACKEND = "openai/gpt-5.3-codex"
BRIDGE_PREFIX_WORDS = 3000
OPENAI_API_KEY = "os.environ/OPENAI_API_KEY"
CACHE_ATTEMPTS = 3

INPUT_RATE = 4e-05
OUTPUT_RATE = 8e-05
CACHE_READ_RATE = 1e-05
CACHE_WRITE_RATE = 5e-05

PRIME_QUESTION = "Reply with the single word ready."
REASONING_QUESTION = "Compute 47*83 - 19*7 step by step, then reply with just the final number."
REASONING_EFFORT = "high"


class _StreamChunk(BaseModel):
    id: str | None = None


def _cache_priced_params(backend: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=backend,
        api_key=OPENAI_API_KEY,
        input_cost_per_token=INPUT_RATE,
        output_cost_per_token=OUTPUT_RATE,
        cache_read_input_token_cost=CACHE_READ_RATE,
        cache_creation_input_token_cost=CACHE_WRITE_RATE,
    )


def _chat_body(
    model: str, content: str, *, stream: bool = False, reasoning_effort: str | None = None
) -> ChatBody:
    return ChatBody(
        model=model,
        messages=[ChatMessage(role="user", content=content)],
        stream=stream,
        max_completion_tokens=4000,
        reasoning_effort=reasoning_effort,
    )


def _require_row(client: SpendClient, request_id: str) -> CostRow:
    row = poll_cost_row(client.proxy, request_id)
    assert row is not None, f"no spend row with a cost breakdown landed for {request_id}"
    return row


def _assert_cache_read_billed(row: CostRow) -> None:
    assert row.breakdown.cache_read_cost is not None and approx_equal(
        row.breakdown.cache_read_cost, row.cache_read_tokens * CACHE_READ_RATE
    ), (
        f"cache_read_cost {row.breakdown.cache_read_cost} != "
        f"{row.cache_read_tokens} cached tokens * {CACHE_READ_RATE}"
    )
    assert_fresh_tokens_billed_at(row, INPUT_RATE)
    assert_total_is_sum_of_components(row)


class TestCacheCostAccounting:
    @pytest.mark.covers("quota_management.spend_tracking.cache_write.bills_cache_creation_rate")
    def test_cache_write_tokens_billed_at_cache_creation_rate(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy, resources, "cache-write-priced", _cache_priced_params(CACHE_WRITE_BACKEND)
        )

        for _ in range(CACHE_ATTEMPTS):
            prompt = f"{cacheable_prefix(unique_marker())}\n{PRIME_QUESTION}"
            chat = unwrap(client.proxy.chat(scoped_key, _chat_body(model, prompt)))
            assert chat.id, f"chat response carried no id: {chat}"
            row = _require_row(client, chat.id)
            if row.cache_creation_tokens > 0:
                break
        else:
            pytest.fail(
                f"OpenAI reported no cache-write tokens across {CACHE_ATTEMPTS} fresh "
                "~2k-token prompts; the cache-write billing path was never exercised"
            )

        assert row.breakdown.cache_creation_cost is not None and approx_equal(
            row.breakdown.cache_creation_cost, row.cache_creation_tokens * CACHE_WRITE_RATE
        ), (
            f"cache_creation_cost {row.breakdown.cache_creation_cost} != "
            f"{row.cache_creation_tokens} cache-write tokens * {CACHE_WRITE_RATE}"
        )
        assert_fresh_tokens_billed_at(row, INPUT_RATE)
        assert_total_is_sum_of_components(row)

    @pytest.mark.covers("quota_management.spend_tracking.cost_breakdown.reports_component_costs")
    def test_cost_breakdown_reports_component_costs(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy, resources, "breakdown-priced", _cache_priced_params(CACHE_READ_BACKEND)
        )

        for _ in range(CACHE_ATTEMPTS):
            prefix = cacheable_prefix(unique_marker())
            unwrap(
                client.proxy.chat(
                    scoped_key,
                    _chat_body(
                        model, f"{prefix}\n{PRIME_QUESTION}", reasoning_effort=REASONING_EFFORT
                    ),
                )
            )
            chat = unwrap(
                client.proxy.chat(
                    scoped_key,
                    _chat_body(
                        model,
                        f"{prefix}\n{REASONING_QUESTION}",
                        reasoning_effort=REASONING_EFFORT,
                    ),
                )
            )
            assert chat.id, f"chat response carried no id: {chat}"
            row = _require_row(client, chat.id)
            if row.cache_read_tokens > 0:
                break
        else:
            pytest.fail(
                f"no cache read landed across {CACHE_ATTEMPTS} prime+read rounds; "
                "the component-cost breakdown was never exercised with cached input"
            )

        usage = chat.usage
        assert usage is not None and usage.completion_tokens_details is not None, (
            f"no completion token details on the measured call: {chat}"
        )
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
        assert reasoning_tokens > 0, f"the reasoning question produced no reasoning tokens: {usage}"

        breakdown = row.breakdown
        assert breakdown.output_cost is not None and approx_equal(
            breakdown.output_cost, (row.completion_tokens or 0) * OUTPUT_RATE
        ), (
            f"output_cost {breakdown.output_cost} != "
            f"{row.completion_tokens} completion tokens * {OUTPUT_RATE}"
        )
        assert breakdown.reasoning_cost is not None and approx_equal(
            breakdown.reasoning_cost, reasoning_tokens * OUTPUT_RATE
        ), (
            f"reasoning_cost {breakdown.reasoning_cost} != "
            f"{reasoning_tokens} reasoning tokens * {OUTPUT_RATE}"
        )
        assert breakdown.reasoning_cost <= (breakdown.output_cost or 0.0) * 1.01, (
            f"reasoning_cost {breakdown.reasoning_cost} exceeds output_cost "
            f"{breakdown.output_cost}; reasoning must be a subset of output"
        )
        _assert_cache_read_billed(row)

    @pytest.mark.covers("quota_management.spend_tracking.stream_cache_read.bills_cache_read_rate")
    def test_streaming_cache_read_billed_at_cache_read_rate(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy, resources, "stream-cache-priced", _cache_priced_params(CACHE_READ_BACKEND)
        )

        for _ in range(CACHE_ATTEMPTS):
            prefix = cacheable_prefix(unique_marker())
            unwrap(client.proxy.chat(scoped_key, _chat_body(model, f"{prefix}\n{PRIME_QUESTION}")))
            result = client.proxy.chat_stream(
                scoped_key,
                _chat_body(model, f"{prefix}\nReply with the single word cached.", stream=True),
            )
            assert result.ok and result.stream_events, (
                f"streamed chat failed (status {result.status_code}): {result.body[:300]}"
            )
            stream_id = _StreamChunk.model_validate_json(result.stream_events[0]).id
            assert stream_id, f"first stream chunk carried no id: {result.stream_events[0][:200]}"
            row = _require_row(client, stream_id)
            if row.cache_read_tokens > 0:
                break
        else:
            pytest.fail(
                f"no cache read landed across {CACHE_ATTEMPTS} prime+stream rounds; "
                "streaming cache-read billing was never exercised"
            )

        _assert_cache_read_billed(row)

    @pytest.mark.covers("quota_management.spend_tracking.messages_bridge.keeps_cache_tokens")
    def test_messages_bridge_keeps_cache_tokens(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy, resources, "bridge-cache-priced", _cache_priced_params(BRIDGE_BACKEND)
        )

        def bridge_call(content: str) -> int:
            response = unwrap(
                client.proxy.messages(
                    scoped_key,
                    AnthropicMessagesBody(
                        model=model,
                        messages=[ChatMessage(role="user", content=content)],
                        max_tokens=4000,
                    ),
                )
            )
            assert response.usage is not None, f"bridged response carried no usage: {response}"
            return response.usage.cache_read_input_tokens or 0

        for _ in range(CACHE_ATTEMPTS):
            prefix = cacheable_prefix(unique_marker(), words=BRIDGE_PREFIX_WORDS)
            bridge_call(f"{prefix}\n{PRIME_QUESTION}")
            if bridge_call(f"{prefix}\nReply with the single word bridged.") > 0:
                break
        else:
            pytest.fail(
                f"no cache read survived {CACHE_ATTEMPTS} bridged prime+read rounds; "
                "cache tokens are not surviving the anthropic-messages -> Responses bridge"
            )

        row = poll_cost_row_where(client.proxy, scoped_key, lambda r: r.cache_read_tokens > 0)
        assert row is not None, (
            "the bridged call reported cached tokens but no spend row for the key "
            "recorded any; the cache tokens were dropped on the way to the bill"
        )
        _assert_cache_read_billed(row)
