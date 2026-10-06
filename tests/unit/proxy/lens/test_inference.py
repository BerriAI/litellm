import json
from collections.abc import Mapping
from math import isclose
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.proxy.lens.inference import (
    Deployment,
    DeploymentParams,
    cache_injection_points,
    completion_charge,
    context_failure,
    exceeds_context,
    model_step,
    output_tokens,
    quote,
    request_messages,
)
from litellm.proxy.lens.models import ModelMessage, ModelRequest
from litellm.types.utils import ModelResponse


def test_missing_optional_price_tiers_use_base_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    litellm.register_model(
        model_cost={
            "openai/lens-base-rate-test": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_output_tokens": 16384,
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "input_cost_per_token_above_200k_tokens": None,
                "output_cost_per_token_above_200k_tokens": None,
                "input_cost_per_token_above_128k_tokens": None,
                "output_cost_per_token_above_128k_tokens": None,
                "input_cost_per_token_above_272k_tokens": None,
                "output_cost_per_token_above_272k_tokens": None,
                "cache_creation_input_token_cost": None,
                "cache_creation_input_token_cost_above_200k_tokens": None,
                "cache_creation_input_token_cost_above_272k_tokens": None,
            }
        }
    )
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-base-rate-test"))
    explicit: Final = Deployment(
        litellm_params=DeploymentParams(
            model="openai/lens-base-rate-test",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            max_tokens=16384,
        )
    )
    assert quote((deployment,), "Answer the question") == quote((explicit,), "Answer the question")


def test_unpriced_model_requires_explicit_rates() -> None:
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-unpriced-test"))
    with pytest.raises(HTTPException) as error:
        quote((deployment,), "Answer the question")
    assert error.value.status_code == 400
    assert "input_cost_per_token" in error.value.detail
    assert "output_cost_per_token" in error.value.detail


def test_custom_priced_model_charges_reported_tokens() -> None:
    deployment: Final = Deployment(
        litellm_params=DeploymentParams(
            model="openai/lens-test", input_cost_per_token=0.001, output_cost_per_token=0.002, max_tokens=16384
        )
    )
    response: Final = ModelResponse(
        model="lens-test", usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    )
    assert completion_charge((deployment,), response, 10) == pytest.approx(0.04)
    assert quote((deployment,), "hello") > 0.04


@pytest.mark.parametrize("capacity", (8192, 65536, 128000))
def test_output_allowance_and_budget_follow_the_models_capacity(capacity: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.lens.inference import output_tokens

    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    litellm.register_model(
        model_cost={
            "openai/lens-capacity-test": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_output_tokens": capacity,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0.001,
            }
        }
    )
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-capacity-test"))
    assert output_tokens(deployment) == capacity
    assert quote((deployment,), "Review") == pytest.approx(capacity * 0.001)


def test_explicit_deployment_output_setting_is_respected() -> None:
    from litellm.proxy.lens.inference import output_tokens

    deployment: Final = Deployment(litellm_params=DeploymentParams(model="custom/model", max_tokens=32000))
    assert output_tokens(deployment) == 32000


def test_shared_context_capacity_leaves_room_for_the_entire_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.lens.inference import output_tokens

    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    litellm.register_model(
        model_cost={
            "openai/lens-shared-context": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_output_tokens": 8192,
                "max_input_tokens": 8192,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0.001,
            }
        }
    )
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-shared-context"))
    short: Final = output_tokens(deployment, "Review this trace")
    long: Final = output_tokens(deployment, "Review this trace " * 500)
    assert 0 < long < short < output_tokens(deployment)
    assert quote((deployment,), "Review this trace " * 500) == pytest.approx(long * 0.001)


def test_unknown_model_capacity_requires_explicit_operator_metadata() -> None:
    from litellm.proxy.lens.inference import ModelCapacity, output_tokens

    params: Final = DeploymentParams(model="openai/lens-unknown-capacity")
    with pytest.raises(HTTPException) as error:
        output_tokens(Deployment(litellm_params=params))
    assert error.value.status_code == 400
    assert "model_info.max_output_tokens" in error.value.detail
    configured: Final = Deployment(litellm_params=params, model_info=ModelCapacity(max_output_tokens=32000))
    assert output_tokens(configured) == 32000


def test_context_preflight_only_rejects_when_every_deployment_is_too_small() -> None:
    from litellm.proxy.lens.inference import ModelCapacity

    params: Final = DeploymentParams(model="openai/lens-configured-context", max_tokens=400)
    short: Final = ModelRequest(prompt="Review", purpose="extract")
    long: Final = ModelRequest(
        prompt="Review",
        purpose="extract",
        messages=(
            ModelMessage(role="user", content="Review"),
            ModelMessage(role="assistant", content="Read the original trace"),
            ModelMessage(role="user", content="Original trace evidence. " * 600),
        ),
    )
    small: Final = Deployment(litellm_params=params, model_info=ModelCapacity(max_input_tokens=1000))
    large: Final = Deployment(litellm_params=params, model_info=ModelCapacity(max_input_tokens=10000))
    assert not exceeds_context((small, large), short)
    assert not exceeds_context((large,), long)
    assert not exceeds_context((large, small), long)
    assert not exceeds_context((small, large), long)
    assert exceeds_context((small,), long)
    unknown: Final = Deployment(litellm_params=params)
    assert not exceeds_context((unknown,), long)
    assert not exceeds_context((small, unknown), long)


def test_provider_context_failure_recognizes_typed_overflow_without_reclassifying_other_errors() -> None:
    from litellm.exceptions import ContextWindowExceededError
    from litellm.proxy._types import ProxyException

    overflow: Final = ContextWindowExceededError(
        message="Provider input limit", model="analysis", llm_provider="openai"
    )
    wrapped: Final = ProxyException(message="redacted", type="invalid_request_error", param=None, code=400)
    wrapped.__cause__ = overflow
    coded: Final = ProxyException(
        message="redacted", type="invalid_request_error", param=None, code=400, openai_code="context_length_exceeded"
    )
    unrelated: Final = ProxyException(
        message="context_length_exceeded appears in user data", type="permission_error", param=None, code=403
    )
    assert context_failure(overflow)
    assert context_failure(wrapped)
    assert context_failure(coded)
    assert not context_failure(unrelated)


def test_a_model_step_records_the_serving_model_and_its_tokens() -> None:
    response: Final = ModelResponse(model="gpt-5.6", usage={"prompt_tokens": 1200, "completion_tokens": 80})
    step: Final = model_step(response, ModelRequest(prompt="review", purpose="extract"), "analysis", 0.02)
    assert (step.model, step.prompt_tokens, step.completion_tokens, step.cost) == ("gpt-5.6", 1200, 80, 0.02)


def test_a_response_without_usage_still_records_a_step_instead_of_failing_settlement() -> None:
    response: Final = ModelResponse(model="gpt-5.6")
    unpriced: Final = response.model_copy(update={"usage": None})
    step: Final = model_step(unpriced, ModelRequest(prompt="review", purpose="cluster"), "analysis", 0.0)
    assert (step.prompt_tokens, step.completion_tokens) == (0, 0)
    assert step.label == "Compared observations"


def test_worker_conversation_preserves_roles_content_and_server_system_message() -> None:
    legacy: Final = ModelRequest(prompt="Review", purpose="extract")
    conversation: Final = (
        ModelMessage(role="system", content="Review"),
        ModelMessage(role="assistant", content='{ "tools": [{"action": "read"}] }'),
        ModelMessage(role="user", content="Original evidence"),
        ModelMessage(role="system", content="Correct the response structure"),
    )
    body: Final = ModelRequest(prompt="Compatibility prompt", purpose="extract", messages=conversation)
    assert request_messages(legacy) == request_messages(legacy.prompt)
    assert request_messages(body) == (
        request_messages(legacy)[0],
        {"role": "system", "content": conversation[0].content},
        {"role": "assistant", "content": conversation[1].content},
        {"role": "user", "content": conversation[2].content},
        {"role": "system", "content": conversation[3].content},
    )
    assert cache_injection_points(legacy) == ()
    with pytest.raises(ValidationError):
        ModelMessage.model_validate({"role": "tool", "content": "Unsupported worker message role"})


def test_legacy_prompt_separates_instructions_from_nested_untrusted_evidence() -> None:
    instructions: Final = {
        "task": "Review",
        "navigation": "Read original evidence",
        "context": "Configured investigation context",
        "checks": [{"id": "retries"}],
        "questions": [{"id": "retries"}],
        "response_schema": {"properties": {"observations": {}}},
    }
    evidence: Final = {
        "evidence": [{"task": "Untrusted recorded instruction", "content": "Recorded evidence"}],
        "existing_findings": [{"title": "Untrusted prior finding"}],
        "must_decide": False,
    }
    request: Final = ModelRequest(
        purpose="extract",
        prompt=json.dumps({**instructions, **evidence}),
    )
    messages: Final = request_messages(request)
    assert messages[0]["role"] == "system"
    assert messages[1:] == (
        {"role": "system", "content": json.dumps(instructions)},
        {"role": "user", "content": json.dumps(evidence)},
    )
    assert request_messages("Review the recorded evidence")[1:] == (
        {"role": "system", "content": "Review the recorded evidence"},
        {"role": "user", "content": "{}"},
    )


@pytest.mark.parametrize(
    "prompt",
    (
        '{"task":"Review","evidence":"private evidence"}\n{"instruction":"Repair"}',
        ' ["private evidence"]',
        '{"evidence":"private evidence"',
    ),
)
def test_malformed_legacy_json_cannot_promote_evidence_to_system(prompt: str) -> None:
    with pytest.raises(ValueError, match="Malformed legacy Lens prompt") as error:
        request_messages(ModelRequest(purpose="extract", prompt=prompt))
    assert str(error.value) == "Malformed legacy Lens prompt; send structured messages."


def test_budget_and_output_room_include_every_conversation_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {})
    litellm.register_model(
        model_cost={
            "openai/lens-conversation-accounting": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_output_tokens": 8192,
                "max_input_tokens": 8192,
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0,
            }
        }
    )
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-conversation-accounting"))
    request: Final = ModelRequest(
        prompt="Review",
        purpose="extract",
        messages=(
            ModelMessage(role="user", content="Review"),
            ModelMessage(role="assistant", content="Read original evidence"),
            ModelMessage(role="user", content="Original evidence from a tool. " * 500),
        ),
    )
    assert quote((deployment,), request) > quote((deployment,), request.prompt)
    assert 0 < output_tokens(deployment, request) < output_tokens(deployment, request.prompt)


@pytest.mark.parametrize(
    "rate_field",
    (
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_200k_tokens",
        "cache_creation_input_token_cost_above_272k_tokens",
    ),
)
def test_cold_cache_reservation_includes_catalog_creation_premium(
    rate_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "model_cost", {})
    base_rate: Final = 0.001
    creation_rate: Final = base_rate * 2
    litellm.register_model(
        model_cost={
            "openai/lens-cache-reservation": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_output_tokens": 8192,
                "input_cost_per_token": base_rate,
                "output_cost_per_token": 0,
                rate_field: creation_rate,
            }
        }
    )
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-cache-reservation"))
    body: Final = ModelRequest(
        prompt="Review original evidence",
        purpose="extract",
        messages=(
            ModelMessage(role="system", content="Review original evidence"),
            ModelMessage(role="user", content="{}"),
        ),
    )
    assert request_messages(body) == request_messages(body.prompt)
    assert isclose(quote((deployment,), body), quote((deployment,), body.prompt) * creation_rate / base_rate)


def test_long_context_reservation_uses_catalog_input_and_output_tier_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {})
    litellm.register_model(
        model_cost={
            "openai/lens-long-context-reservation": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_output_tokens": 8192,
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "input_cost_per_token_above_272k_tokens": 0.003,
                "output_cost_per_token_above_272k_tokens": 0.005,
            }
        }
    )
    deployment: Final = Deployment(litellm_params=DeploymentParams(model="openai/lens-long-context-reservation"))
    worst_case: Final = Deployment(
        litellm_params=DeploymentParams(
            model="openai/lens-long-context-reservation", input_cost_per_token=0.003, output_cost_per_token=0.005
        )
    )
    assert quote((deployment,), "Original evidence") == quote((worst_case,), "Original evidence")


class CacheBlock(BaseModel):
    text: str
    prompt_cache_breakpoint: Mapping[str, str] | None = None


class CacheMessage(BaseModel):
    role: str
    content: str | tuple[CacheBlock, ...]


def test_cache_hook_marks_prior_write_boundary_when_the_conversation_grows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", {})
    litellm.register_model(
        model_cost={
            "openai/lens-cache-boundary-test": {
                "litellm_provider": "openai",
                "mode": "chat",
                "supports_prompt_cache_breakpoint": True,
            }
        }
    )
    messages: Final = (
        ModelMessage(role="system", content="Static task"),
        ModelMessage(role="user", content="Initial evidence"),
        ModelMessage(role="assistant", content="Read another span"),
        ModelMessage(role="user", content="First tool response"),
        ModelMessage(role="assistant", content="Read remaining evidence"),
        ModelMessage(role="user", content="Second tool response"),
    )
    for size in (2, 4, 6):
        body = ModelRequest(prompt="Static task", purpose="extract", messages=messages[:size])
        parsed = TypeAdapter(tuple[str, tuple[CacheMessage, ...], Mapping[str, object]]).validate_python(
            AnthropicCacheControlHook().get_chat_completion_prompt(  # pyright: ignore[reportUnknownMemberType]  # shared hook exposes legacy untyped parameter dictionaries
                model="openai/lens-cache-boundary-test",
                messages=list(request_messages(body)),
                non_default_params={
                    "cache_control_injection_points": list(cache_injection_points(body)),
                    "custom_llm_provider": "openai",
                    "api_base": "https://api.openai.com/v1",
                },
                prompt_id=None,
                prompt_variables=None,
                dynamic_callback_params={},
            )
        )
        for index in (1, max(1, size - 2), size):
            content = parsed[1][index].content
            assert not isinstance(content, str)
            assert content[-1].prompt_cache_breakpoint == {"mode": "explicit"}
            assert content[-1].text == body.messages[index - 1].content


def test_parallel_reservations_wait_without_charging_or_falsely_exhausting_budget() -> None:
    from functools import reduce

    from litellm.proxy.lens.inference import reserve_amount, settle_amount
    from litellm.proxy.lens.models import BudgetReservation
    from tests.unit.proxy.lens.test_state import lens

    initial: Final = lens().model_copy(update={"spent": 45})
    reservations: Final = tuple(
        BudgetReservation(id=str(index), job_id="run", amount=10, month=initial.budget_month) for index in range(6)
    )
    held: Final = reduce(reserve_amount, reservations[:5], initial)
    assert held.spent == 45
    assert sum(item.amount for item in held.reservations) == 50
    assert reserve_amount(held, reservations[5]) is held
    settled: Final = settle_amount(held, "0", 0.25, None)
    assert settled.spent == 45.25
    assert len(settled.reservations) == 4
    assert settle_amount(settled, "0", 0.25, None) is settled
    assert reservations[5] in reserve_amount(settled, reservations[5]).reservations
    with pytest.raises(HTTPException, match="needs up to"):
        reserve_amount(initial, reservations[0].model_copy(update={"amount": 60}))


def test_expired_reservations_do_not_hold_budget_and_late_settlement_still_charges() -> None:
    from datetime import timedelta

    from litellm.proxy.lens.inference import reserve_amount, settle_amount
    from litellm.proxy.lens.models import BudgetReservation
    from tests.unit.proxy.lens.test_state import NOW, lens

    stale: Final = BudgetReservation(id="stale", job_id="run", amount=90, month=lens().budget_month, expires_at=NOW)
    initial: Final = lens().model_copy(update={"reservations": (stale,)})
    incoming: Final = stale.model_copy(update={"id": "current", "expires_at": NOW + timedelta(minutes=5)})
    waiting: Final = reserve_amount(initial, incoming, NOW - timedelta(seconds=1))
    assert waiting is initial
    admitted: Final = reserve_amount(initial, incoming, NOW)
    assert incoming in admitted.reservations
    assert admitted.spent == 0
    settled: Final = settle_amount(admitted, "stale", 0.25, None)
    assert settled.spent == 0.25
    assert settled.reservations == (incoming,)


def test_abandoned_reservations_are_pruned_after_late_settlement_retention() -> None:
    from datetime import timedelta

    from litellm.proxy.lens.inference import reserve_amount
    from litellm.proxy.lens.models import BudgetReservation
    from tests.unit.proxy.lens.test_state import NOW, lens

    stale: Final = BudgetReservation(
        id="stale", job_id="run", amount=90, month=lens().budget_month, expires_at=NOW - timedelta(days=1)
    )
    recent: Final = stale.model_copy(update={"id": "recent", "expires_at": NOW})
    incoming: Final = stale.model_copy(update={"id": "active", "expires_at": NOW + timedelta(minutes=5)})
    admitted: Final = reserve_amount(lens().model_copy(update={"reservations": (stale, recent)}), incoming, NOW)
    assert admitted.reservations == (recent, incoming)
    assert admitted.spent == 0


def test_renewed_model_call_keeps_budget_reserved_until_it_finishes_or_its_lease_expires() -> None:
    from datetime import timedelta

    from litellm.proxy.lens.inference import BUDGET_LEASE, renew_reservation, reserve_amount, settle_amount
    from litellm.proxy.lens.models import BudgetReservation
    from tests.unit.proxy.lens.test_state import NOW, lens

    active: Final = BudgetReservation(
        id="active", job_id="run", amount=90, month=lens().budget_month, expires_at=NOW + BUDGET_LEASE
    )
    other: Final = active.model_copy(update={"id": "other", "amount": 1})
    renewed: Final = renew_reservation(
        lens().model_copy(update={"reservations": (active, other)}), active.id, NOW + BUDGET_LEASE / 2
    )
    incoming: Final = active.model_copy(update={"id": "incoming", "amount": 20})
    assert renewed.reservations[1] == other
    assert renewed.spent == 0
    assert reserve_amount(renewed, incoming, NOW + BUDGET_LEASE + timedelta(seconds=1)) is renewed
    assert incoming in reserve_amount(renewed, incoming, NOW + BUDGET_LEASE * 2).reservations
    settled: Final = settle_amount(renewed, active.id, 0.25, None)
    assert settled.reservations == (other,)
    assert settled.spent == 0.25


@pytest.mark.parametrize("missing", (False, True))
def test_renewal_does_not_resurrect_expired_or_released_budget(missing: bool) -> None:
    from litellm.proxy.lens.inference import renew_reservation
    from litellm.proxy.lens.models import BudgetReservation
    from tests.unit.proxy.lens.test_state import NOW, lens

    expired: Final = BudgetReservation(id="expired", job_id="run", amount=90, month=lens().budget_month, expires_at=NOW)
    initial: Final = lens().model_copy(update={"reservations": () if missing else (expired,)})
    with pytest.raises(HTTPException) as error:
        renew_reservation(initial, expired.id, NOW)
    assert error.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("completed", "model_failed", "lease_lost", "cancelled"))
async def test_model_call_and_budget_renewal_finish_together(outcome: str) -> None:
    import asyncio

    from litellm.proxy.lens.inference import model_with_renewal

    model_started: Final = asyncio.Event()
    renewal_started: Final = asyncio.Event()
    model_finished: Final = asyncio.Event()
    renewal_finished: Final = asyncio.Event()
    release: Final = asyncio.Event()
    response: Final = (ModelResponse(model="analysis"), 0.25)

    async def model() -> tuple[ModelResponse, float | None]:
        try:
            model_started.set()
            await renewal_started.wait()
            if outcome == "model_failed":
                raise HTTPException(503, "Model failed")
            if outcome != "completed":
                await release.wait()
            return response
        finally:
            model_finished.set()

    async def renewal() -> None:
        try:
            renewal_started.set()
            await model_started.wait()
            if outcome == "lease_lost":
                raise HTTPException(503, "Reservation lost")
            await release.wait()
        finally:
            renewal_finished.set()

    request: Final = asyncio.create_task(model_with_renewal(model(), renewal()))
    if outcome == "cancelled":
        await model_started.wait()
        await renewal_started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    elif outcome == "completed":
        assert await request is response
    else:
        with pytest.raises(HTTPException) as error:
            await request
        assert error.value.detail == ("Model failed" if outcome == "model_failed" else "Reservation lost")
    assert model_finished.is_set()
    assert renewal_finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("timed_out", (False, True))
async def test_renewal_preserves_the_original_failure_while_request_cleanup_is_pending(timed_out: bool) -> None:
    import asyncio

    from litellm.proxy.lens.inference import (
        BUDGET_LEASE,
        model_with_renewal,
        renew_reservation,
        reserve_amount,
        reserved_budget,
    )
    from litellm.proxy.lens.models import BudgetReservation
    from litellm.proxy.lens.repository import LensRepository
    from tests.unit.proxy.lens.test_endpoints import ResultDatabase
    from tests.unit.proxy.lens.test_state import NOW, lens

    db: Final = ResultDatabase(lens())
    repo: Final = LensRepository(db)
    hold: Final = BudgetReservation(
        id="active", job_id="run", amount=90, month=db.stored.budget_month, expires_at=NOW + BUDGET_LEASE
    )
    admitted: Final = asyncio.Event()
    unwinding: Final = asyncio.Event()
    renewed: Final = asyncio.Event()
    stopped: Final = asyncio.Event()
    failure: Final = (
        TimeoutError("Model timed out") if timed_out else HTTPException(400, "Provider rejected the request")
    )

    async def model() -> tuple[ModelResponse, float | None]:
        try:
            async with reserved_budget(repo, "lens", hold.id, lambda e: reserve_amount(e, hold, NOW), admitted):
                raise failure
        except HTTPException:
            unwinding.set()
            await renewed.wait()
            raise

    async def renewal() -> None:
        try:
            await unwinding.wait()
            await repo.update("lens", lambda e: renew_reservation(e, hold.id, NOW))
            renewed.set()
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with pytest.raises(HTTPException) as error:
        await model_with_renewal(model(), renewal())
    assert (error.value.__cause__ is failure) if timed_out else (error.value is failure)
    assert error.value.status_code == (504 if timed_out else 400)
    assert stopped.is_set()
    assert db.stored.reservations == (hold,)


@pytest.mark.asyncio
async def test_completed_paid_response_survives_simultaneous_renewal_failure() -> None:
    from litellm.proxy.lens.inference import model_with_renewal

    response: Final = (ModelResponse(model="analysis"), 0.25)

    async def model() -> tuple[ModelResponse, float | None]:
        return response

    async def renewal() -> None:
        raise HTTPException(503, "Reservation lost")

    assert await model_with_renewal(model(), renewal()) is response


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
@pytest.mark.parametrize("cleanup", ("success", "missing", "unavailable"))
async def test_failed_budget_cleanup_preserves_the_original_request_error(cancelled: bool, cleanup: str) -> None:
    import asyncio
    from collections.abc import AsyncGenerator
    from contextlib import asynccontextmanager

    from litellm.proxy.lens.inference import release_failed_reservation
    from litellm.proxy.lens.models import BudgetReservation
    from litellm.proxy.lens.repository import Database, LensRepository, Row
    from tests.unit.proxy.lens.test_state import lens

    hold: Final = BudgetReservation(id="paid", job_id="job", amount=10, month=lens().budget_month)

    class CleanupDatabase:
        def __init__(self) -> None:
            self.stored = lens().model_copy(update={"reservations": (hold,)})

        @asynccontextmanager
        async def transaction(self) -> AsyncGenerator[Database]:
            if cleanup == "unavailable":
                raise OSError("Database is unavailable")
            yield self

        async def query_raw(self, query: str, *args: object) -> tuple[Row, ...]:
            if cleanup == "missing":
                return ()
            if query.startswith("SELECT data FROM"):
                return (Row(data=self.stored.model_dump(mode="json")),)
            assert isinstance(args[0], str)
            self.stored = type(self.stored).model_validate_json(args[0])
            return (Row(data=1),)

        async def execute_raw(self, query: str, *args: object) -> int:
            raise AssertionError("No checkpoint writes expected")

    db: Final = CleanupDatabase()
    failure: Final = asyncio.CancelledError() if cancelled else HTTPException(400, "Provider rejected the request")
    with pytest.raises(type(failure)) as error:
        async with release_failed_reservation(LensRepository(db), "lens", hold.id):
            raise failure
    assert error.value is failure
    assert db.stored.reservations == (() if cleanup == "success" else (hold,))
    assert db.stored.spent == 0
