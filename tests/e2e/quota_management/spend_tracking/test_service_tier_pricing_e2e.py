"""Live e2e: a service_tier request bills every component at the tier's own rates.

Pins the tier-billing fixes (#35923, #35925): a priority-tier call must price
input and output at the deployment's `*_priority` rates, including the reasoning
tokens inside output (the shipped bug billed reasoning at the default-tier rate),
and the spend row must record the tier the bill was computed on.

The deployment carries custom base AND priority rates, each distinct, so a bill
computed from the wrong tier (or a mix) cannot match the expected numbers. The
prompt is a fresh unique marker per run, keeping cached tokens out of the math.
The response's own `service_tier` echo is asserted first: if OpenAI ever declined
priority processing and served the default tier, the test fails there instead of
producing a vacuous rate comparison. Reasoning is requested explicitly with
`reasoning_effort`, so the reasoning-rate assertion rests on a parameter the test
sets rather than on whatever the model happens to do by default.

The streaming cases pin the served-tier contract: OpenAI stamps the tier it actually
used on every stream chunk, and that echo is what the caller sees and what the bill
must be computed on. The request sets no service_tier, so the only place the tier
can come from is the provider's response. The spend row must record the served tier
and price input at that tier's rate, and every chunk the proxy relays must carry the
same service_tier the provider sent.
"""

import json

import pytest
from cost_rows import (
    approx_equal,
    assert_fresh_tokens_billed_at,
    assert_total_is_sum_of_components,
    poll_cost_row,
    poll_cost_row_where,
    register_priced_model,
)
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import (
    AnthropicMessagesBody,
    ChatBody,
    ChatMessage,
    ChatStreamOptions,
    LiteLLMParamsBody,
    ResponsesStreamBody,
)
from pydantic import BaseModel
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

BACKEND = "openai/gpt-5.6-luna"
OPENAI_API_KEY = "os.environ/OPENAI_API_KEY"
STREAM_BACKEND = f"openai/{CHEAP_OPENAI_MODEL}"

INPUT_RATE = 4e-05
OUTPUT_RATE = 8e-05
PRIORITY_INPUT_RATE = 6e-05
PRIORITY_OUTPUT_RATE = 1.6e-04

REASONING_EFFORT = "high"

TIER_INPUT_RATES = {"default": INPUT_RATE, "priority": PRIORITY_INPUT_RATE}


class _StreamChunk(BaseModel):
    id: str | None = None
    service_tier: str | None = None


class _CompletedResponseObject(BaseModel):
    id: str | None = None
    service_tier: str | None = None


class _ResponsesStreamEvent(BaseModel):
    type: str | None = None
    response: _CompletedResponseObject | None = None


class _MessagesStreamEvent(BaseModel):
    type: str | None = None


def _stream_chunks(events: list[str]) -> list[_StreamChunk]:
    return [_StreamChunk.model_validate_json(event) for event in events if event.strip() != "[DONE]"]


def _served_tier(chunks: list[_StreamChunk]) -> str:
    tiers = {chunk.service_tier for chunk in chunks if chunk.service_tier}
    assert len(tiers) == 1, (
        f"the relayed stream carried {tiers or 'no'} service tier(s) across {len(chunks)} chunks; OpenAI stamps "
        "the served tier on every chat chunk, so exactly one tier must reach the caller"
    )
    return tiers.pop()


class TestServiceTierPricing:
    @pytest.mark.covers("quota_management.spend_tracking.service_tier.bills_tier_rates")
    def test_priority_tier_bills_priority_rates(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy,
            resources,
            "tier-priced",
            LiteLLMParamsBody(
                model=BACKEND,
                api_key=OPENAI_API_KEY,
                input_cost_per_token=INPUT_RATE,
                output_cost_per_token=OUTPUT_RATE,
                input_cost_per_token_priority=PRIORITY_INPUT_RATE,
                output_cost_per_token_priority=PRIORITY_OUTPUT_RATE,
            ),
        )

        chat = unwrap(
            client.proxy.chat(
                scoped_key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=(
                                f"{unique_marker()} Compute 47*83 - 19*7 step by step, "
                                "then reply with just the final number."
                            ),
                        )
                    ],
                    max_completion_tokens=4000,
                    service_tier="priority",
                    reasoning_effort=REASONING_EFFORT,
                ),
            )
        )
        assert chat.service_tier == "priority", (
            f"OpenAI served tier {chat.service_tier!r} instead of priority; tier billing was never exercised"
        )
        assert chat.id, f"chat response carried no id: {chat}"

        row = poll_cost_row(client.proxy, chat.id)
        assert row is not None, f"no spend row with a cost breakdown landed for {chat.id}"
        breakdown = row.breakdown

        assert breakdown.service_tier == "priority", (
            f"the bill records pricing basis {breakdown.service_tier!r}, not priority"
        )

        assert_fresh_tokens_billed_at(row, PRIORITY_INPUT_RATE)
        assert breakdown.output_cost is not None and approx_equal(
            breakdown.output_cost, (row.completion_tokens or 0) * PRIORITY_OUTPUT_RATE
        ), (
            f"output_cost {breakdown.output_cost} != {row.completion_tokens} tokens * priority rate "
            f"{PRIORITY_OUTPUT_RATE} (base rate would give {(row.completion_tokens or 0) * OUTPUT_RATE})"
        )

        usage = chat.usage
        assert usage is not None and usage.completion_tokens_details is not None, (
            f"no completion token details on the priority call: {chat}"
        )
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
        assert reasoning_tokens > 0, f"the reasoning question produced no reasoning tokens: {usage}"
        assert breakdown.reasoning_cost is not None and approx_equal(
            breakdown.reasoning_cost, reasoning_tokens * PRIORITY_OUTPUT_RATE
        ), (
            f"reasoning_cost {breakdown.reasoning_cost} != {reasoning_tokens} reasoning tokens * "
            f"priority rate {PRIORITY_OUTPUT_RATE} (the default-tier rate would give "
            f"{reasoning_tokens * OUTPUT_RATE})"
        )

        assert_total_is_sum_of_components(row)

    @pytest.mark.covers("quota_management.spend_tracking.service_tier_stream.records_served_tier")
    def test_streamed_call_records_and_bills_the_served_tier(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy,
            resources,
            "tier-priced-stream",
            LiteLLMParamsBody(
                model=BACKEND,
                api_key=OPENAI_API_KEY,
                input_cost_per_token=INPUT_RATE,
                output_cost_per_token=OUTPUT_RATE,
                input_cost_per_token_priority=PRIORITY_INPUT_RATE,
                output_cost_per_token_priority=PRIORITY_OUTPUT_RATE,
            ),
        )

        result = client.proxy.chat_stream(
            scoped_key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=f"{unique_marker()} reply with one word")],
                max_completion_tokens=64,
                stream=True,
            ),
        )
        assert result.ok and result.stream_events, (
            f"streamed chat failed (status {result.status_code}): {result.body[:300]}"
        )
        chunks = _stream_chunks(result.stream_events)
        served_tier = _served_tier(chunks)
        assert served_tier in TIER_INPUT_RATES, f"no custom rate registered for served tier {served_tier!r}"
        stream_id = chunks[0].id
        assert stream_id, f"first stream chunk carried no id: {result.stream_events[0][:200]}"

        row = poll_cost_row(client.proxy, stream_id)
        assert row is not None, f"no spend row with a cost breakdown landed for {stream_id}"
        assert row.breakdown.service_tier == served_tier, (
            f"the provider served tier {served_tier!r} on every chunk but the bill records "
            f"pricing basis {row.breakdown.service_tier!r}"
        )
        assert_fresh_tokens_billed_at(row, TIER_INPUT_RATES[served_tier])
        assert_total_is_sum_of_components(row)

    @pytest.mark.covers("llm.chat_completions.openai.service_tier.stream.echoes_served_tier")
    def test_every_streamed_chunk_carries_the_served_tier(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy, resources, "tier-echo-stream", LiteLLMParamsBody(model=STREAM_BACKEND, api_key=OPENAI_API_KEY)
        )
        result = client.proxy.chat_stream(
            scoped_key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=f"{unique_marker()} reply with one word")],
                max_completion_tokens=64,
                stream=True,
                stream_options=ChatStreamOptions(include_usage=True),
            ),
        )
        assert result.ok and result.stream_events, (
            f"streamed chat failed (status {result.status_code}): {result.body[:300]}"
        )
        chunks = _stream_chunks(result.stream_events)
        served_tier = _served_tier(chunks)
        missing = [
            json.loads(event) for event, chunk in zip(result.stream_events, chunks) if chunk.service_tier is None
        ]
        assert not missing, (
            f"{len(missing)} of {len(chunks)} relayed chunks dropped the provider's service_tier "
            f"{served_tier!r}: {missing}"
        )

    @pytest.mark.covers("quota_management.spend_tracking.service_tier_stream.responses_records_served_tier")
    def test_responses_stream_records_the_served_tier(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy,
            resources,
            "tier-responses-stream",
            LiteLLMParamsBody(model=STREAM_BACKEND, api_key=OPENAI_API_KEY),
        )

        result = client.proxy.responses_stream(
            scoped_key,
            ResponsesStreamBody(model=model, input=f"{unique_marker()} reply with one word"),
        )
        assert result.ok and result.stream_events, (
            f"streamed responses call failed (status {result.status_code}): {result.body[:300]}"
        )

        events = [_ResponsesStreamEvent.model_validate_json(event) for event in result.stream_events]
        completed = next((event for event in reversed(events) if event.type == "response.completed"), None)
        assert completed is not None and completed.response is not None, (
            f"no response.completed event in the stream: {[e.type for e in events]}"
        )
        served_tier = completed.response.service_tier
        assert served_tier, f"response.completed carried no service_tier: {completed.response}"
        assert served_tier in TIER_INPUT_RATES, f"no custom rate registered for served tier {served_tier!r}"
        assert completed.response.id, f"response.completed carried no id: {completed.response}"

        row = poll_cost_row(client.proxy, completed.response.id)
        assert row is not None, f"no spend row with a cost breakdown landed for {completed.response.id}"
        assert row.breakdown.service_tier == served_tier, (
            f"response.completed served tier {served_tier!r} but the bill records "
            f"pricing basis {row.breakdown.service_tier!r}"
        )

    @pytest.mark.covers("quota_management.spend_tracking.service_tier_stream.messages_records_served_tier")
    def test_messages_stream_records_the_served_tier(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy,
            resources,
            "tier-messages-stream",
            LiteLLMParamsBody(model=STREAM_BACKEND, api_key=OPENAI_API_KEY),
        )

        result = client.proxy.messages_stream(
            scoped_key,
            AnthropicMessagesBody(
                model=model,
                messages=[ChatMessage(role="user", content=f"{unique_marker()} reply with one word")],
                max_tokens=64,
                stream=True,
            ),
        )
        assert result.ok and result.stream_events, (
            f"streamed messages call failed (status {result.status_code}): {result.body[:300]}"
        )

        events = [_MessagesStreamEvent.model_validate_json(event) for event in result.stream_events]
        assert any(event.type == "message_delta" for event in events), (
            f"the anthropic stream emitted no message_delta: {[e.type for e in events]}"
        )

        row = poll_cost_row_where(client.proxy, scoped_key, lambda r: r.spend is not None and r.spend > 0)
        assert row is not None, f"no spend row with a cost breakdown landed for the streamed messages call on {model}"
        served_tier = row.breakdown.service_tier
        assert served_tier in TIER_INPUT_RATES and served_tier is not None, (
            "the anthropic wire format carries no service_tier, so the bill is the only record of "
            f"the tier OpenAI served; the row recorded pricing basis {served_tier!r}"
        )
