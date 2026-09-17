from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Final, Literal

import pytest
from cost_rows import CostRow, approx_equal, poll_cost_row
from e2e_config import settle_propagation, unique_marker
from e2e_http import NoBody, StreamingResponse, unwrap
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    ChatResponse,
    KeyDeleteBody,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ModelDeleteBody,
)
from pydantic import BaseModel, RootModel
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

INPUT_RATE: Final = 0.00005
OUTPUT_RATE: Final = 0.0001
DISCOUNT: Final = 0.25
MARGIN_PERCENT: Final = 0.1
MARGIN_FIXED: Final = 0.0005


class _ConfigPatchResponse(BaseModel):
    status: str
    values: dict[str, float | dict[str, float]]


class _DiscountConfig(RootModel[dict[str, float]]):
    pass


class _MarginConfig(RootModel[dict[str, float | dict[str, float]]]):
    pass


class _DiscountConfigResponse(BaseModel):
    values: dict[str, float]


class _MarginConfigResponse(BaseModel):
    values: dict[str, float | dict[str, float]]


class _BedrockParams(BaseModel):
    guardrail: Literal["bedrock"] = "bedrock"
    mode: Literal["pre_call"] = "pre_call"
    default_on: bool = False
    guardrailIdentifier: str
    guardrailVersion: str


class _GuardrailSpec(BaseModel):
    guardrail_name: str
    litellm_params: _BedrockParams


class _GuardrailCreate(BaseModel):
    guardrail: _GuardrailSpec


class _GuardrailCreateResponse(BaseModel):
    guardrail_id: str


def _register_bedrock_guardrail(
    client: SpendClient,
    resources: ResourceManager,
    name: str,
    identifier: str,
    version: str,
) -> None:
    guardrail_id: Final = unwrap(
        client.proxy.transport.post(
            "/guardrails",
            headers=client.proxy.transport.master,
            json=_GuardrailCreate(
                guardrail=_GuardrailSpec(
                    guardrail_name=name,
                    litellm_params=_BedrockParams(
                        guardrailIdentifier=identifier,
                        guardrailVersion=version,
                    ),
                )
            ),
            response_type=_GuardrailCreateResponse,
        )
    ).guardrail_id
    settle_propagation(time.monotonic())
    resources.defer(lambda: _delete_guardrail(client, guardrail_id))


def _delete_guardrail(client: SpendClient, guardrail_id: str) -> None:
    unwrap(
        client.proxy.transport.delete(
            f"/guardrails/{guardrail_id}",
            headers=client.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )
    )


def _guarded_chat(client: SpendClient, key: str, model: str, name: str) -> StreamingResponse:
    return client.proxy.transport.send(
        "/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=ChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=f"reply with a short greeting {unique_marker()}")],
            max_tokens=16,
            guardrails=[name],
        ),
    )


def _set_discount(client: SpendClient, values: dict[str, float]) -> None:
    unwrap(
        client.proxy.transport.patch(
            "/config/cost_discount_config",
            headers=client.proxy.transport.master,
            json=_DiscountConfig(values),
            response_type=_ConfigPatchResponse,
        )
    )


def _set_margin(client: SpendClient, values: dict[str, float | dict[str, float]]) -> None:
    unwrap(
        client.proxy.transport.patch(
            "/config/cost_margin_config",
            headers=client.proxy.transport.master,
            json=_MarginConfig(values),
            response_type=_ConfigPatchResponse,
        )
    )


def _get_discount(client: SpendClient) -> dict[str, float]:
    return unwrap(
        client.proxy.transport.get(
            "/config/cost_discount_config",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=_DiscountConfigResponse,
        )
    ).values


def _get_margin(client: SpendClient) -> dict[str, float | dict[str, float]]:
    return unwrap(
        client.proxy.transport.get(
            "/config/cost_margin_config",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=_MarginConfigResponse,
        )
    ).values


def _register_model(client: SpendClient, resources: ResourceManager, prefix: str) -> str:
    model: Final = f"{prefix}-{unique_marker()}"
    model_id: Final = client.proxy.create_model(
        model,
        LiteLLMParamsBody(
            model="openai/gpt-4o-mini",
            api_key="os.environ/OPENAI_API_KEY",
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
        ),
    )
    resources.defer(lambda: _delete_model(client, model_id))
    return model


def _delete_model(client: SpendClient, model_id: str) -> None:
    unwrap(
        client.proxy.transport.post(
            "/model/delete",
            headers=client.proxy.transport.master,
            json=ModelDeleteBody(id=model_id),
            response_type=NoBody,
        )
    )


def _delete_key(client: SpendClient, key: str) -> None:
    unwrap(
        client.proxy.transport.post(
            "/key/delete",
            headers=client.proxy.transport.master,
            json=KeyDeleteBody(keys=[key]),
            response_type=NoBody,
        )
    )


def _base_cost(row: CostRow, prompt_tokens: int, completion_tokens: int) -> float:
    assert prompt_tokens > 0 and completion_tokens > 0
    assert row.prompt_tokens == prompt_tokens and row.completion_tokens == completion_tokens
    base_cost: Final = prompt_tokens * INPUT_RATE + completion_tokens * OUTPUT_RATE
    assert row.breakdown.original_cost is not None
    assert approx_equal(row.breakdown.original_cost, base_cost)
    return base_cost


@pytest.fixture
def strict_resources(client: SpendClient) -> Iterator[ResourceManager]:
    manager: Final = ResourceManager(client=client.proxy, strict_cleanup=True)
    manager.init()
    yield manager
    manager.teardown()


@pytest.fixture
def scoped_key(client: SpendClient, strict_resources: ResourceManager) -> str:
    key: Final = client.proxy.generate_key(KeyGenerateBody(user_id="e2e-test-user"))
    strict_resources.defer(lambda: _delete_key(client, key))
    return key


@pytest.fixture
def restored_pricing_config(client: SpendClient) -> Iterator[None]:
    discount: Final = _get_discount(client)
    margin: Final = _get_margin(client)
    try:
        _set_discount(client, {})
        _set_margin(client, {})
        yield
    finally:
        try:
            _set_discount(client, discount)
        finally:
            _set_margin(client, margin)


class TestPricingConfigSpend:
    @pytest.mark.covers(
        "quota_management.spend_tracking.discount_config.logs_cost",
        exercised_on=["chat_completions"],
    )
    def test_configured_discount_reaches_persisted_spend_row(
        self,
        client: SpendClient,
        strict_resources: ResourceManager,
        scoped_key: str,
        restored_pricing_config: None,
    ) -> None:
        _set_discount(client, {"openai": DISCOUNT})
        model: Final = _register_model(client, strict_resources, "discount-priced")
        chat: Final = unwrap(client.chat(scoped_key, model, f"reply with one word {unique_marker()}", max_tokens=16))
        assert chat.id and chat.usage and chat.usage.prompt_tokens and chat.usage.completion_tokens

        row: Final = poll_cost_row(client.proxy, chat.id)
        assert row is not None
        base_cost: Final = _base_cost(row, chat.usage.prompt_tokens, chat.usage.completion_tokens)
        breakdown: Final = row.breakdown
        assert breakdown.discount_percent is not None and approx_equal(breakdown.discount_percent, DISCOUNT)
        assert breakdown.discount_amount is not None and approx_equal(breakdown.discount_amount, base_cost * DISCOUNT)
        assert row.spend is not None and approx_equal(row.spend, base_cost * (1 - DISCOUNT))

    @pytest.mark.covers(
        "quota_management.spend_tracking.margin_config.logs_cost",
        exercised_on=["chat_completions"],
    )
    def test_configured_margin_reaches_persisted_spend_row(
        self,
        client: SpendClient,
        strict_resources: ResourceManager,
        scoped_key: str,
        restored_pricing_config: None,
    ) -> None:
        _set_margin(client, {"openai": {"percentage": MARGIN_PERCENT, "fixed_amount": MARGIN_FIXED}})
        model: Final = _register_model(client, strict_resources, "margin-priced")
        chat: Final = unwrap(client.chat(scoped_key, model, f"reply with one word {unique_marker()}", max_tokens=16))
        assert chat.id and chat.usage and chat.usage.prompt_tokens and chat.usage.completion_tokens

        row: Final = poll_cost_row(client.proxy, chat.id)
        assert row is not None
        base_cost: Final = _base_cost(row, chat.usage.prompt_tokens, chat.usage.completion_tokens)
        breakdown: Final = row.breakdown
        expected_margin: Final = base_cost * MARGIN_PERCENT + MARGIN_FIXED
        assert breakdown.margin_percent is not None and approx_equal(breakdown.margin_percent, MARGIN_PERCENT)
        assert breakdown.margin_fixed_amount is not None and approx_equal(breakdown.margin_fixed_amount, MARGIN_FIXED)
        assert breakdown.margin_total_amount is not None and approx_equal(breakdown.margin_total_amount, expected_margin)
        assert row.spend is not None and approx_equal(row.spend, base_cost + expected_margin)

    @pytest.mark.covers(
        "quota_management.spend_tracking.guardrail_cost.logs_cost",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_guardrail_cost_reaches_persisted_spend_row(
        self,
        client: SpendClient,
        strict_resources: ResourceManager,
        scoped_key: str,
        restored_pricing_config: None,
    ) -> None:
        identifier: Final = os.environ["BEDROCK_GUARDRAIL_IDENTIFIER"]
        version: Final = os.environ["BEDROCK_GUARDRAIL_VERSION"]
        name: Final = f"e2e-bedrock-cost-{unique_marker()}"
        _register_bedrock_guardrail(client, strict_resources, name, identifier, version)
        model: Final = _register_model(client, strict_resources, "guardrail-priced")
        result: Final = _guarded_chat(client, scoped_key, model, name)
        assert result.ok, f"guarded request failed with {result.status_code}: {result.body[:400]}"
        assert name in {value.strip() for value in result.headers.get("x-litellm-applied-guardrails", "").split(",")}

        chat: Final = ChatResponse.model_validate_json(result.body)
        assert chat.id and chat.usage and chat.usage.prompt_tokens and chat.usage.completion_tokens
        row: Final = poll_cost_row(client.proxy, chat.id)
        assert row is not None
        base_cost: Final = _base_cost(row, chat.usage.prompt_tokens, chat.usage.completion_tokens)
        breakdown: Final = row.breakdown
        assert breakdown.guardrail_cost is not None and breakdown.guardrail_cost > 0
        assert breakdown.total_cost is not None and approx_equal(breakdown.total_cost, base_cost + breakdown.guardrail_cost)
        assert row.spend is not None and approx_equal(row.spend, breakdown.total_cost)
