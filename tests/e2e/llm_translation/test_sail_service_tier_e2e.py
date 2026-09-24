"""Live e2e: the gateway translates `service_tier` into Sail's `metadata.completion_window`.

Sail is a JSON-registered OpenAI-compatible provider that has no `service_tier`.
A caller's `flex` or `balanced` becomes `metadata.completion_window` on the wire
and `default` becomes `asap` (#42840). The proof a real provider gives: a window
the model does not offer comes back as Sail's own 400 naming the rewritten
field, a served call carries no `service_tier` at all, and its
`x-litellm-response-cost` is that window's cost-map rates times the usage Sail
reported. Every rate is read off the proxy's cost map at run time, never pinned.
"""

from __future__ import annotations

import re
from typing import Final

import pytest
from pydantic import BaseModel

from completion_window_pricing import (
    SAIL_API_KEY,
    TierRates,
    cheapest_flex_only_model,
    cheapest_window_priced_model,
    within_rel,
)
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

MAX_COMPLETION_TOKENS: Final = 64


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


def _sail_deployment(client: PassthroughClient, resources: ResourceManager, prefix: str, backend: str) -> str:
    model = f"{prefix}-{unique_marker()}"
    model_id = client.proxy.create_model(model, LiteLLMParamsBody(model=backend, api_key=SAIL_API_KEY))
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model


def _chat_total(rates: TierRates, chat: ChatResponse) -> float:
    usage = chat.usage
    assert usage is not None and usage.prompt_tokens is not None and usage.completion_tokens is not None, (
        f"sail chat {chat.id} carried no usage: {chat}"
    )
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return rates.bill(
        prompt_tokens=usage.prompt_tokens,
        cached_tokens=(prompt_details.cached_tokens or 0) if prompt_details else 0,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=(completion_details.reasoning_tokens or 0) if completion_details else 0,
    )[4]


def _served_chat(client: PassthroughClient, key: str, model: str, service_tier: str | None) -> StreamingResponse:
    outcome = client.proxy.transport.send(
        "/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=ChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=f"{unique_marker()} Reply with the single word ok.")],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            service_tier=service_tier,
        ),
    )
    assert outcome.status_code == 200, (
        f"sail chat with service_tier={service_tier!r} failed {outcome.status_code}: {outcome.body}"
    )
    return outcome


class TestSailServiceTier:
    @pytest.mark.covers("llm.chat_completions.sail.service_tier.nonstream.works")
    def test_chat_prices_each_service_tier_as_its_window_and_echoes_none(
        self, client: PassthroughClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        priced = cheapest_window_priced_model(client.proxy.model_cost_map())
        model = _sail_deployment(client, resources, "sail-tier", priced.model)

        outcomes = {tier: _served_chat(client, scoped_key, model, tier) for tier in ("flex", "balanced", None)}
        chats = {tier: ChatResponse.model_validate_json(outcome.body) for tier, outcome in outcomes.items()}
        assert {tier: chat.service_tier for tier, chat in chats.items()} == {
            "flex": None,
            "balanced": None,
            None: None,
        }, f"sail has no service_tier, yet the responses reported {[chat.service_tier for chat in chats.values()]}"

        costs = {tier: outcome.response_cost for tier, outcome in outcomes.items()}
        expected = {
            "flex": _chat_total(priced.flex, chats["flex"]),
            "balanced": _chat_total(priced.balanced, chats["balanced"]),
            None: _chat_total(priced.asap, chats[None]),
        }
        assert costs.keys() == expected.keys() and all(within_rel(costs[t], expected[t]) for t in expected), (
            f"x-litellm-response-cost per tier {costs} != each window's rates times its usage {expected} "
            f"(rates {priced})"
        )
        assert len(set(costs.values())) == 3, f"the three windows did not price three different totals: {costs}"

    @pytest.mark.covers("llm.responses.sail.service_tier.nonstream.works")
    def test_responses_extra_body_completion_window_prices_that_window(
        self, client: PassthroughClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        priced = cheapest_window_priced_model(client.proxy.model_cost_map())
        model = _sail_deployment(client, resources, "sail-resp", priced.model)

        outcome = client.proxy.transport.send(
            "/v1/responses",
            headers=client.proxy.transport.bearer(scoped_key),
            json=_WindowedResponsesBody(
                model=model,
                input=f"{unique_marker()} Reply with the single word ok.",
                max_output_tokens=MAX_COMPLETION_TOKENS,
                extra_body=_WindowExtraBody(metadata=_WindowMetadata(completion_window="balanced")),
            ),
        )
        assert outcome.status_code == 200, f"sail /v1/responses failed {outcome.status_code}: {outcome.body}"
        response = _ResponsesObject.model_validate_json(outcome.body)
        expected = priced.balanced.bill(
            prompt_tokens=response.usage.input_tokens,
            cached_tokens=response.usage.input_tokens_details.cached_tokens,
            completion_tokens=response.usage.output_tokens,
            reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
        )[4]
        assert within_rel(outcome.response_cost, expected), (
            f"{response.id} priced {outcome.response_cost}, expected {expected} from {response.usage} at the "
            f"balanced rates {priced.balanced}"
        )

    @pytest.mark.covers("llm.chat_completions.sail.service_tier.nonstream.works")
    def test_flex_only_model_serves_flex_and_refuses_the_windows_other_tiers_become(
        self, client: PassthroughClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        flex_only = cheapest_flex_only_model(client.proxy.model_cost_map())
        model = _sail_deployment(client, resources, "sail-flex", flex_only.model)

        served = _served_chat(client, scoped_key, model, "flex")
        chat = ChatResponse.model_validate_json(served.body)
        expected = _chat_total(flex_only.flex, chat)
        assert within_rel(served.response_cost, expected), (
            f"{chat.id} priced {served.response_cost}, expected {expected} at the flex rates {flex_only.flex}"
        )

        windows = {"default": "asap", "balanced": "balanced"}
        refusals = {tier: _refused_window(client, scoped_key, model, tier) for tier in windows}
        assert {tier: bool(re.search(rf'"{windows[tier]}"', seen)) for tier, seen in refusals.items()} == {
            "default": True,
            "balanced": True,
        }, f"service_tier must reach sail as completion_window {windows}; sail saw: {refusals}"


def _refused_window(client: PassthroughClient, key: str, model: str, service_tier: str) -> str:
    """Sail's own message for a completion_window the model does not offer, once the
    gateway has relayed it as a 400 naming the rewritten field."""
    outcome = client.proxy.transport.send(
        "/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=ChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=f"{unique_marker()} Reply with the single word ok.")],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            service_tier=service_tier,
        ),
    )
    assert outcome.status_code == 400, (
        f"expected sail's 400 for service_tier={service_tier!r}, got {outcome.status_code}: {outcome.body}"
    )
    error = _ErrorBody.model_validate_json(outcome.body).error
    relayed = re.fullmatch(
        r"litellm\.BadRequestError: SailException - (?P<upstream>.+)\n\n"
        rf"LiteLLM: model group '{re.escape(model)}' failed with the error above\. No fallback was attempted\.",
        error.message,
        re.DOTALL,
    )
    assert relayed is not None, f"the error is not sail's message relayed for {model}: {error.message!r}"
    assert (error.type, error.param, error.code) == ("invalid_request_error", "metadata.completion_window", "400"), (
        f"sail refused {error.param!r}, not the completion_window the gateway rewrote service_tier into: {error}"
    )
    return relayed["upstream"]
