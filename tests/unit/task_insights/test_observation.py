from datetime import datetime, timezone

import pytest

from litellm.models.task_insights import TASK_INSIGHTS_ANALYSIS_TAG, TaskObservation, UnobservedDelivery
from litellm.task_insights.observation import (
    MAX_CONTEXT_TEXT_CHARS,
    MAX_REQUEST_TEXT_CHARS,
    observe,
    observe_applied_deliveries,
    task_insights_enabled,
)

_AT = datetime(2026, 11, 11, 9, 30, tzinfo=timezone.utc)


def _delivery(payload: dict[str, object] | None = None, **fields: object) -> UnobservedDelivery:
    return UnobservedDelivery.model_validate(
        {
            "delivery_id": "d1",
            "request_id": "r1",
            "is_batch": False,
            "accounting_payload": {
                "startTime": "2026-11-02T12:00:00+00:00",
                "call_type": "anthropic_messages",
                "spend": 0.4,
                "model": "claude-sonnet",
                "status": "success",
                "request_duration_ms": 1200,
                **(payload or {}),
            },
            "model_config_id": "config-sonnet",
            "logged_model": "claude-sonnet",
            **fields,
        }
    )


def test_a_failed_attempt_keeps_its_charge_and_drops_its_latency():
    observation = observe(_delivery({"status": "failure", "spend": 0.25, "request_duration_ms": 30000}), _AT)

    assert (observation.spend, observation.latency_ms, observation.status) == (0.25, None, "failure")


@pytest.mark.parametrize("spend", [None, "0.4", float("nan"), float("inf"), True])
def test_a_missing_or_invalid_charge_stays_unknown(spend):
    payload = {"spend": spend} if spend is not None else {}
    delivery = _delivery(payload)
    if spend is None:
        delivery = delivery.model_copy(
            update={"accounting_payload": {k: v for k, v in delivery.accounting_payload.items() if k != "spend"}}
        )

    assert observe(delivery, _AT).spend is None


@pytest.mark.parametrize("latency", [-5, float("nan"), "fast"])
def test_an_invalid_latency_stays_unknown(latency):
    assert observe(_delivery({"request_duration_ms": latency}), _AT).latency_ms is None


def test_a_delivery_billed_on_another_model_keeps_an_unknown_deployment():
    assert observe(_delivery(logged_model="claude-opus"), _AT).model_config_id is None
    assert observe(_delivery(), _AT).model_config_id == "config-sonnet"
    assert observe(_delivery(model_config_id=""), _AT).model_config_id is None


@pytest.mark.parametrize(
    "payload",
    [
        {"request_tags": f'["{TASK_INSIGHTS_ANALYSIS_TAG}"]'},
        {"request_tags": [TASK_INSIGHTS_ANALYSIS_TAG, "other"]},
        {"metadata": {"internal_call_origin": "llm_as_a_judge"}},
    ],
)
def test_analysis_and_internal_calls_are_excluded_without_storing_their_text(payload):
    observation = observe(_delivery(payload, turns=({"role": "user", "content": "classify these"},)), _AT)

    assert observation.excluded is True
    assert (observation.request_text, observation.text_hash) == (None, None)


def test_tool_output_turns_are_skipped_and_the_first_instruction_becomes_context():
    turns = (
        {"role": "system", "content": "You are a coding agent"},
        {"role": "user", "content": "Add pagination to the orders endpoint"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "edit", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "file contents"}]},
        {"role": "user", "content": [{"type": "text", "text": "Now validate the page size\x00"}]},
    )

    observation = observe(_delivery(turns=turns), _AT)

    assert observation.request_text == "Now validate the page size"
    assert observation.context_text == "Add pagination to the orders endpoint"


def test_identical_instructions_share_a_hash_and_different_context_does_not():
    first = observe(_delivery(turns=({"role": "user", "content": "Fix the build"},)), _AT)
    replay = observe(_delivery(delivery_id="d2", turns=({"role": "user", "content": "Fix the build"},)), _AT)
    other = observe(
        _delivery(turns=({"role": "user", "content": "Ship it"}, {"role": "user", "content": "Fix the build"})), _AT
    )

    assert first.text_hash == replay.text_hash
    assert first.text_hash != other.text_hash


def test_responses_input_text_and_bounds():
    observation = observe(_delivery(input_text="x" * (MAX_REQUEST_TEXT_CHARS + 50)), _AT)
    long_context = observe(
        _delivery(turns=({"role": "user", "content": "y" * 5000}, {"role": "user", "content": "short"})), _AT
    )

    assert len(observation.request_text or "") == MAX_REQUEST_TEXT_CHARS
    assert len(long_context.context_text or "") == MAX_CONTEXT_TEXT_CHARS


def test_every_delivery_yields_an_observation_even_without_text_or_time():
    observation = observe(_delivery({"startTime": "not a time"}), _AT)

    assert observation.start_time == _AT
    assert (observation.request_text, observation.text_hash) == (None, None)


class _Store:
    def __init__(self, batches: list[tuple[UnobservedDelivery, ...]]) -> None:
        self.batches = batches
        self.inserted: list[TaskObservation] = []
        self.prefixes: list[str | None] = []

    async def unobserved_deliveries(self, limit: int, request_id_prefix: str | None = None):
        self.prefixes.append(request_id_prefix)
        return self.batches.pop(0) if self.batches else ()

    async def insert_observations(self, observations):
        self.inserted.extend(observations)
        return len(observations)


@pytest.mark.asyncio
async def test_projection_stops_at_a_short_batch_and_counts_inserts():
    store = _Store([(_delivery(), _delivery(delivery_id="d2")), (_delivery(delivery_id="d3"),), (_delivery(),)])

    observed = await observe_applied_deliveries(
        store, now=lambda: _AT, batch_size=2, max_batches=10, request_id_prefix="p"
    )

    assert observed == 3
    assert [item.delivery_id for item in store.inserted] == ["d1", "d2", "d3"]
    assert store.prefixes == ["p", "p"]


@pytest.mark.asyncio
async def test_projection_is_bounded_by_max_batches():
    store = _Store([(_delivery(),)] * 5)

    assert await observe_applied_deliveries(store, now=lambda: _AT, batch_size=1, max_batches=2) == 2


@pytest.mark.parametrize(
    ("value", "enabled"), [("true", True), ("1", True), (" YES ", True), ("false", False), ("", False)]
)
def test_feature_flag(monkeypatch, value, enabled):
    monkeypatch.setenv("LITELLM_TASK_INSIGHTS_ENABLED", value)

    assert task_insights_enabled() is enabled


def test_a_failed_management_request_is_observed_but_never_counted_as_traffic():
    management = observe(
        _delivery({"call_type": "/management/v1/users/me", "status": "failure", "spend": 0.0, "model": ""}), _AT
    )
    failed_inference = observe(_delivery({"status": "failure", "spend": 0.005}), _AT)

    assert (management.excluded, management.inference) == (True, False)
    assert (failed_inference.excluded, failed_inference.inference, failed_inference.spend) == (False, True, 0.005)
