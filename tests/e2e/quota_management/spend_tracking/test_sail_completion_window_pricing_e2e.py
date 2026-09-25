from __future__ import annotations

import re
from typing import Final

import pytest
from pydantic import BaseModel, RootModel

from completion_window_pricing import (
    SAIL_API_KEY,
    TierRates,
    all_within_rel,
    cheapest_flex_only_model,
    cheapest_window_priced_model,
    within_rel,
)
from cost_rows import CostBreakdownRow, CostRow, poll_cost_row, register_priced_model
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody, SpendLogsParams
from proxy_client import ProxyClient
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

OPENAI_BACKEND: Final = "openai/gpt-5.5"
OPENAI_API_KEY: Final = "os.environ/OPENAI_API_KEY"
MAX_COMPLETION_TOKENS: Final = 64
"""OpenAI validates `service_tier` against its own enum and rejects `balanced`
(observed 2026-09-24: "Supported values are: 'auto', 'default', 'fast', 'flex',
and 'priority'"), so the balanced leg on OpenAI can only prove the gateway forwards
the tier untouched and bills nothing for the refusal."""


class _WindowMetadata(BaseModel):
    completion_window: str


class _WindowExtraBody(BaseModel):
    metadata: _WindowMetadata


class _WindowedResponsesBody(BaseModel):
    model: str
    input: str
    max_output_tokens: int
    extra_body: _WindowExtraBody
    cache: dict[str, bool] = {"no-cache": True}


class _ResponsesInputDetails(BaseModel):
    cached_tokens: int = 0


class _ResponsesOutputDetails(BaseModel):
    reasoning_tokens: int = 0


class _ResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    input_tokens_details: _ResponsesInputDetails = _ResponsesInputDetails()
    output_tokens_details: _ResponsesOutputDetails = _ResponsesOutputDetails()


class _ResponsesObject(BaseModel):
    id: str
    usage: _ResponsesUsage


class _ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str


class _ErrorBody(BaseModel):
    error: _ErrorDetail


class _StatusRowMetadata(BaseModel):
    status: str | None = None
    cost_breakdown: CostBreakdownRow | None = None


class _StatusRow(BaseModel):
    request_id: str | None = None
    spend: float | None = None
    metadata: _StatusRowMetadata | None = None


class _StatusRows(RootModel[list[_StatusRow]]):
    pass


class _ChatUsage(BaseModel):
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int


def _chat_usage(chat: ChatResponse) -> _ChatUsage:
    usage = chat.usage
    assert usage is not None and usage.prompt_tokens is not None and usage.completion_tokens is not None, (
        f"chat response {chat.id} carried no usage: {chat}"
    )
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return _ChatUsage(
        prompt_tokens=usage.prompt_tokens,
        cached_tokens=(prompt_details.cached_tokens or 0) if prompt_details else 0,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=(completion_details.reasoning_tokens or 0) if completion_details else 0,
    )


def _billed(row: CostRow) -> tuple[float, float, float, float, float]:
    breakdown = row.breakdown
    return (
        breakdown.input_cost or 0.0,
        breakdown.output_cost or 0.0,
        breakdown.cache_read_cost or 0.0,
        breakdown.reasoning_cost or 0.0,
        breakdown.total_cost or 0.0,
    )


def _assert_billed_at(row: CostRow, rates: TierRates, usage: _ChatUsage, *, tier: str | None, window: str) -> None:
    expected = rates.bill(
        prompt_tokens=usage.prompt_tokens,
        cached_tokens=usage.cached_tokens,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )
    assert (row.prompt_tokens, row.completion_tokens) == (usage.prompt_tokens, usage.completion_tokens), (
        f"{window} row {row.request_id} logged {row.prompt_tokens}/{row.completion_tokens} tokens, the response "
        f"reported {usage.prompt_tokens}/{usage.completion_tokens}"
    )
    assert all_within_rel(_billed(row), expected), (
        f"{window} row {row.request_id} billed (input, output, cache_read, reasoning, total) {_billed(row)}, "
        f"expected {expected} from {usage} at the {window} rates {rates}"
    )
    assert within_rel(row.spend, expected[4]), (
        f"{window} row {row.request_id} spend {row.spend} != its total_cost {expected[4]}"
    )
    assert row.breakdown.service_tier == tier, (
        f"{window} row {row.request_id} records pricing basis {row.breakdown.service_tier!r}, expected {tier!r}"
    )


def _landed_row(proxy: ProxyClient, request_id: str) -> CostRow:
    row = poll_cost_row(proxy, request_id)
    assert row is not None, f"no spend row with a cost breakdown landed for {request_id}"
    return row


def _row_shape(row: _StatusRow) -> tuple[float | None, str | None, bool]:
    metadata = row.metadata
    return (row.spend, metadata.status if metadata else None, bool(metadata and metadata.cost_breakdown))


def _poll_key_rows(proxy: ProxyClient, key: str) -> tuple[_StatusRow, ...]:
    landed = proxy.poll_logs_for_key(key)
    assert landed, f"no spend row landed for the key before the {proxy.poll_timeout}s deadline"
    return tuple(
        unwrap(
            proxy.transport.get(
                "/spend/logs",
                headers=proxy.transport.master,
                params=SpendLogsParams(api_key=key),
                response_type=_StatusRows,
            )
        ).root
    )


def _sail_chat(proxy: ProxyClient, key: str, model: str, service_tier: str | None) -> ChatResponse:
    chat = unwrap(
        proxy.chat(
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=f"{unique_marker()} Reply with the single word ok.")],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                service_tier=service_tier,
            ),
        )
    )
    assert chat.id, f"sail chat with service_tier={service_tier!r} carried no id: {chat}"
    return chat


class TestSailCompletionWindowPricing:
    @pytest.mark.covers("quota_management.spend_tracking.completion_window.bills_window_rates")
    def test_each_service_tier_bills_its_windows_cost_map_rates(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        priced = cheapest_window_priced_model(client.proxy.model_cost_map())
        model = register_priced_model(
            client.proxy, resources, "sail-windows", LiteLLMParamsBody(model=priced.model, api_key=SAIL_API_KEY)
        )

        flex = _sail_chat(client.proxy, scoped_key, model, "flex")
        balanced = _sail_chat(client.proxy, scoped_key, model, "balanced")
        asap = _sail_chat(client.proxy, scoped_key, model, None)
        assert (flex.service_tier, balanced.service_tier, asap.service_tier) == (None, None, None), (
            "sail has no service_tier, yet the responses carried "
            f"{(flex.service_tier, balanced.service_tier, asap.service_tier)}"
        )

        flex_row = _landed_row(client.proxy, flex.id or "")
        balanced_row = _landed_row(client.proxy, balanced.id or "")
        asap_row = _landed_row(client.proxy, asap.id or "")
        _assert_billed_at(flex_row, priced.flex, _chat_usage(flex), tier="flex", window="flex")
        _assert_billed_at(balanced_row, priced.balanced, _chat_usage(balanced), tier="balanced", window="balanced")
        _assert_billed_at(asap_row, priced.asap, _chat_usage(asap), tier=None, window="asap")
        totals = (flex_row.spend, balanced_row.spend, asap_row.spend)
        assert len(set(totals)) == 3, f"the three windows did not bill three different totals: {totals}"

    @pytest.mark.covers("quota_management.spend_tracking.completion_window.bills_window_rates")
    def test_responses_completion_window_in_extra_body_bills_balanced_rates(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        priced = cheapest_window_priced_model(client.proxy.model_cost_map())
        model = register_priced_model(
            client.proxy, resources, "sail-responses", LiteLLMParamsBody(model=priced.model, api_key=SAIL_API_KEY)
        )
        window = "balanced"

        response = unwrap(
            client.proxy.transport.post(
                "/v1/responses",
                headers=client.proxy.transport.bearer(scoped_key),
                json=_WindowedResponsesBody(
                    model=model,
                    input=f"{unique_marker()} Reply with the single word ok.",
                    max_output_tokens=MAX_COMPLETION_TOKENS,
                    extra_body=_WindowExtraBody(metadata=_WindowMetadata(completion_window=window)),
                ),
                response_type=_ResponsesObject,
            )
        )
        row = _landed_row(client.proxy, response.id)
        usage = _ChatUsage(
            prompt_tokens=response.usage.input_tokens,
            cached_tokens=response.usage.input_tokens_details.cached_tokens,
            completion_tokens=response.usage.output_tokens,
            reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
        )
        _assert_billed_at(row, priced.balanced, usage, tier=window, window=window)

    @pytest.mark.covers("quota_management.spend_tracking.completion_window.rejected_window_bills_nothing")
    def test_flex_only_model_refuses_default_tier_and_bills_nothing(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        flex_only = cheapest_flex_only_model(client.proxy.model_cost_map())
        model = register_priced_model(
            client.proxy, resources, "sail-flexonly", LiteLLMParamsBody(model=flex_only.model, api_key=SAIL_API_KEY)
        )

        outcome = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=f"{unique_marker()} Reply with the single word ok.")],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                service_tier="default",
            ),
        )
        assert outcome.status_code == 400, f"expected the provider's 400, got {outcome.status_code}: {outcome.body}"
        assert outcome.response_cost == 0, f"a refused request was priced at {outcome.response_cost}"
        error = _ErrorBody.model_validate_json(outcome.body).error
        relayed = re.fullmatch(
            r"litellm\.BadRequestError: SailException - (?P<upstream>.+)\n\n"
            rf"LiteLLM: model group '{re.escape(model)}' failed with the error above\. No fallback was attempted\.",
            error.message,
            re.DOTALL,
        )
        assert relayed is not None, f"the error is not the provider's message relayed for {model}: {error.message!r}"
        assert (error.type, error.param, error.code) == (
            "invalid_request_error",
            "metadata.completion_window",
            "400",
        ), (
            f"the provider refused {error.param!r}, not the completion_window the gateway rewrote service_tier "
            f"into: {error}"
        )
        assert re.search(r'"asap"', relayed["upstream"]), (
            f"service_tier=default must reach sail as completion_window asap; the provider saw: {relayed['upstream']}"
        )

        rows = _poll_key_rows(client.proxy, scoped_key)
        assert [_row_shape(row) for row in rows] == [(0.0, "failure", False)], (
            f"the refused request left billable spend rows under the key: {rows}"
        )
        assert client.proxy.key_info(scoped_key).spend == 0.0, "the refused request added to the key's spend"


class TestOpenAIKeepsItsOwnServiceTier:
    @pytest.mark.covers("quota_management.spend_tracking.service_tier.bills_served_tier")
    def test_openai_bills_the_tier_it_reports_and_refuses_balanced_unbilled(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        cost_map = client.proxy.model_cost_map()
        entry = cost_map[OPENAI_BACKEND] if OPENAI_BACKEND in cost_map else cost_map[OPENAI_BACKEND.split("/", 1)[1]]
        assert (
            entry.input_cost_per_token is not None
            and entry.output_cost_per_token is not None
            and entry.cache_read_input_token_cost is not None
        ), f"{OPENAI_BACKEND} has no base rates in the cost map: {entry}"
        base = TierRates(
            input=entry.input_cost_per_token,
            output=entry.output_cost_per_token,
            cache_read=entry.cache_read_input_token_cost,
        )
        model = register_priced_model(
            client.proxy, resources, "openai-tiers", LiteLLMParamsBody(model=OPENAI_BACKEND, api_key=OPENAI_API_KEY)
        )
        prompt = f"{unique_marker()} Reply with the single word ok."

        refused = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=prompt)],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                service_tier="balanced",
            ),
        )
        assert refused.status_code == 400, f"expected OpenAI's 400, got {refused.status_code}: {refused.body}"
        assert refused.response_cost == 0, f"a refused request was priced at {refused.response_cost}"
        refused_error = _ErrorBody.model_validate_json(refused.body).error
        assert (refused_error.type, refused_error.param, refused_error.code) == (
            "invalid_request_error",
            "service_tier",
            "400",
        ), f"OpenAI must receive service_tier itself, untouched by the sail rewrite: {refused_error}"

        served = unwrap(
            client.proxy.chat(
                scoped_key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=prompt)],
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                ),
            )
        )
        assert served.id and served.service_tier, f"OpenAI reported no service_tier on the untiered call: {served}"
        row = _landed_row(client.proxy, served.id)
        _assert_billed_at(row, base, _chat_usage(served), tier=served.service_tier, window="openai default")

        rows = _poll_key_rows(client.proxy, scoped_key)
        assert sorted((key_row.request_id == served.id, *_row_shape(key_row)) for key_row in rows) == [
            (False, 0.0, "failure", False),
            (True, row.spend, None, True),
        ], f"expected one unbilled failure row and one billed row for {served.id} under the key: {rows}"
