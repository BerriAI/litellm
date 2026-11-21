import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import pytest
import pytest_asyncio
from pydantic import TypeAdapter

import litellm
import litellm.proxy.proxy_server as ps
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.integrations.custom_logger_v2.rust import RustLogger
from litellm.litellm_core_utils.llm_cost_calc.finalization import FinalizedCost, finalize_cost
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.native_database import ServiceLoggingEvents
from litellm.proxy.db.rust_spend_log_writer import install_rust_spend_log_writer, uninstall_rust_spend_log_writer
from litellm.proxy.hooks.rust_spend_logger import (
    _SPEND_LOG_CALL,  # pyright: ignore[reportPrivateUsage]  # the host reads the call it serves from here
    SpendLogHost,
    bill_failure_in_rust,
    rust_spend_loggers,
    spend_logger,
)
from litellm.proxy.spend_tracking.budget_reservation import BudgetCounter
from litellm.proxy.spend_tracking.customer_cost_margin import (
    apply_customer_cost_margin,
    snapshotless_cost_with_customer_margin,
)
from litellm.proxy.spend_tracking.native_budgets import reservation_inputs
from litellm.proxy.spend_tracking.rust_spend_log_inputs import SpendLogCall, spend_log_success_input
from litellm.proxy.utils import BilledFailedAttempt, PrismaClient
from litellm.rust_bridge._native import DatabaseEndpoint, NativeDatabase, NativeSpendCounters, RustSpendLogWriter
from litellm.types.integrations.custom_logger_v2 import Outcome
from tests.unit.proxy.db.test_rust_spend_log_writer import FakePrismaClient

_SETTINGS: Final = TypeAdapter(Mapping[str, object])
_UNREACHABLE: Final = "postgresql://litellm:secret@db.invalid:5432/proxy?pool_timeout=1"
_KEYED: Final[Mapping[str, object]] = {
    "user_api_key": "hashed-key",
    "user_api_key_alias": "alias",
    "user_api_key_user_id": "user-1",
    "user_api_key_team_id": "team-1",
    "user_api_key_org_id": "org-1",
}


@dataclass
class AfterWrite:
    calls: list[Mapping[str, object]]

    async def record(self, **call: object) -> None:  # kwargs-ok: records the post-write step's keyword arguments
        self.calls.append(call)


@dataclass(frozen=True, slots=True)
class Proxy:
    client: FakePrismaClient
    writer: RustSpendLogWriter
    logger: RustLogger
    after_write: AfterWrite

    async def rows(self, count: int = 1) -> Sequence[Mapping[str, object]]:
        for _ in range(300):
            if len(self.writer) >= count:
                break
            await asyncio.sleep(0.01)
        return [json.loads(row) for row in self.writer.dequeue(100)]


@pytest_asyncio.fixture(loop_scope="function")
async def proxy(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Proxy]:
    monkeypatch.setattr(PrismaClient, "spend_log_queue_bytes", 0)
    database: Final = await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: None),
        await DatabaseEndpoint.resolve(_UNREACHABLE, 1),
        None,
    )
    client: Final = FakePrismaClient()
    writer: Final = await install_rust_spend_log_writer(client, database, {"use_spend_logs_ledger": True})  # pyright: ignore[reportArgumentType]  # the fake stands in for the Prisma client
    assert writer is not None
    handle: Final = ps.spend_counter_store.handle
    assert isinstance(handle, NativeSpendCounters)
    after_write: Final = AfterWrite([])
    monkeypatch.setattr(ps, "prisma_client", client)
    monkeypatch.setattr(ps, "general_settings", {"use_spend_logs_ledger": True})
    monkeypatch.setattr(ps.proxy_logging_obj.slack_alerting_instance, "alerting", [])
    monkeypatch.setattr(ps.proxy_logging_obj.db_spend_update_writer, "record_after_ledger_row", after_write.record)
    logger: Final = spend_logger(writer, handle)
    monkeypatch.setattr(litellm, "callbacks", [logger])
    try:
        yield Proxy(client, writer, logger, after_write)
    finally:
        await uninstall_rust_spend_log_writer(client)  # pyright: ignore[reportArgumentType]  # the fake stands in for the Prisma client
        ps.spend_counter_store.flush_local()


async def _complete(metadata: Mapping[str, object]) -> None:
    _ = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello there",
        metadata=dict(metadata),
    )


def _auth(budget_reservation: dict[str, object] | None = None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed-key",
        user_id="user-1",
        team_id="team-1",
        request_route="/v1/responses",
        budget_reservation=budget_reservation,
    )


def _metadata(row: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = row["metadata"]
    assert isinstance(metadata, str)
    decoded: Final = json.loads(metadata)
    assert isinstance(decoded, dict)
    return decoded


@pytest.mark.asyncio
async def test_a_priced_completion_queues_one_ledger_row_and_runs_the_post_write_steps(proxy: Proxy) -> None:
    await _complete(_KEYED)

    rows: Final = await proxy.rows()

    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["team_id"] == "team-1"
    assert isinstance(rows[0]["spend"], float) and rows[0]["spend"] > 0
    assert rows[0]["accounting_delivery_id"]
    for _ in range(100):
        if proxy.after_write.calls:
            break
        await asyncio.sleep(0.01)
    assert len(proxy.after_write.calls) == 1
    call: Final = proxy.after_write.calls[0]
    assert call["team_id"] == "team-1"
    assert call["spend_logs_kept"] is True
    assert call["claimed"] is False
    assert call["payload"]["request_id"] == rows[0]["request_id"]  # pyright: ignore[reportIndexIssue]  # the recorded payload dict
    assert call["payload"]["startTime"] == rows[0]["startTime"]  # pyright: ignore[reportIndexIssue]  # the recorded payload dict
    assert "T" in call["payload"]["startTime"]  # pyright: ignore[reportIndexIssue, reportOperatorIssue]  # the tool usage date takes the text before T


@pytest.mark.asyncio
async def test_a_success_writes_the_settled_reservation_back(proxy: Proxy) -> None:
    reservation: Final[dict[str, object]] = {"reserved_cost": 0.5, "entries": []}

    await _complete({**_KEYED, "user_api_key_budget_reservation": reservation})
    _ = await proxy.rows()
    for _ in range(300):
        if "finalized" in reservation:
            break
        await asyncio.sleep(0.01)

    assert reservation.get("finalized") is True


@pytest.mark.asyncio
async def test_a_failure_writes_the_released_reservation_back(proxy: Proxy) -> None:
    auth: Final = _auth(budget_reservation={"reserved_cost": 0.5, "entries": []})

    await bill_failure_in_rust(auth, ValueError("refused"), {"model": "gpt-4o", "metadata": {}})

    assert auth.budget_reservation is not None
    assert auth.budget_reservation.get("finalized") is True


@pytest.mark.asyncio
async def test_a_rejected_websocket_frame_queues_a_failure_row(proxy: Proxy) -> None:
    await bill_failure_in_rust(
        _auth(), ValueError("guardrail refused the frame"), {"model": "gpt-4o", "call_type": "aresponses"}
    )

    rows: Final = await proxy.rows()

    assert len(rows) == 1
    assert rows[0]["status"] == "failure"
    assert rows[0]["team_id"] == "team-1"


@pytest.mark.asyncio
async def test_a_failure_row_carries_the_callers_traceback(proxy: Proxy) -> None:
    _ = await ps.proxy_logging_obj.post_call_failure_hook(
        request_data={"model": "gpt-4o", "metadata": {}},
        original_exception=ValueError("upstream failed"),
        user_api_key_dict=_auth(),
        traceback_str="the caller's traceback",
    )

    rows: Final = await proxy.rows()

    assert len(rows) == 1
    error_information: Final = _metadata(rows[0])["error_information"]
    assert isinstance(error_information, dict)
    assert str(error_information.get("traceback", "")).startswith("the caller's traceback")


@pytest.mark.asyncio
async def test_a_stored_failure_body_shows_the_failure_stamps_and_released_reservation(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ps, "general_settings", {"use_spend_logs_ledger": True, "store_prompts_in_spend_logs": True})
    reservation: Final[dict[str, object]] = {"reserved_cost": 0.5, "entries": []}
    metadata: Final[dict[str, object]] = {"user_api_key_budget_reservation": reservation}
    _ = await ps.proxy_logging_obj.post_call_failure_hook(
        request_data={
            "model": "gpt-4o",
            "metadata": metadata,
            "proxy_server_request": {"body": {"model": "gpt-4o", "metadata": metadata}},
        },
        original_exception=ValueError("upstream failed"),
        user_api_key_dict=_auth(budget_reservation=reservation),
    )

    rows: Final = await proxy.rows()

    stored: Final = json.loads(str(rows[0]["proxy_server_request"]))["metadata"]
    assert stored["status"] == "failure"
    assert stored["error_information"]["error_class"] == "ValueError"
    assert stored["user_api_key_budget_reservation"]["finalized"] is True


_FAILED_ATTEMPT_USAGE: Final = litellm.Usage(prompt_tokens=24, completion_tokens=0, total_tokens=24)


def _recovered_stream_failure() -> Mapping[str, object]:
    return {"combined_usage_object": _FAILED_ATTEMPT_USAGE, "response_cost": 0.125}


def _guardrail_only_failure(cost_in_spend: bool) -> Mapping[str, object]:
    entry: Final = {"guardrail_name": "priced", "guardrail_cost": 0.3, "guardrail_cost_in_spend": cost_in_spend}
    return {"litellm_metadata": {"standard_logging_guardrail_information": [entry]}}


def _retry_only_failure() -> Mapping[str, object]:
    return {
        "combined_usage_object": _FAILED_ATTEMPT_USAGE,
        "response_cost": 0.0,
        "billed_failed_attempts": (
            BilledFailedAttempt(
                attempted_retries=0, deployment_id=None, usage=_FAILED_ATTEMPT_USAGE, response_cost=0.25
            ),
        ),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_error_logs", [False, True])
@pytest.mark.parametrize(
    ("failure", "charges"),
    [
        pytest.param(_recovered_stream_failure(), (0.125,), id="recovered-stream"),
        pytest.param(_retry_only_failure(), (0.0, 0.25), id="retry-only"),
        pytest.param(_guardrail_only_failure(cost_in_spend=True), (0.3,), id="guardrail-only"),
        pytest.param(_guardrail_only_failure(cost_in_spend=False), (0.0,), id="report-only-guardrail"),
        pytest.param({"combined_usage_object": _FAILED_ATTEMPT_USAGE, "response_cost": 0.0}, (0.0,), id="no-cost"),
    ],
)
async def test_an_unfinalized_failure_keeps_its_ledger_charge_and_logs_only_while_error_logs_are_on(
    proxy: Proxy,
    monkeypatch: pytest.MonkeyPatch,
    disable_error_logs: bool,
    failure: Mapping[str, object],
    charges: tuple[float, ...],
) -> None:
    monkeypatch.setattr(
        ps, "general_settings", {"use_spend_logs_ledger": True, "disable_error_logs": disable_error_logs}
    )

    await bill_failure_in_rust(
        _auth(),
        ValueError("the stream broke"),
        {"model": "gpt-4o", "metadata": {}, "litellm_call_id": "call-1", **failure},
    )

    expected: Final = () if disable_error_logs and not any(charge > 0 for charge in charges) else charges
    rows: Final = [_SETTINGS.validate_json(row) for row in proxy.writer.dequeue(100)]
    assert [_SETTINGS.validate_python(row["accounting_payload"])["spend"] for row in rows] == pytest.approx(
        list(expected)
    )
    assert len({row["request_id"] for row in rows}) == len(rows)
    assert all(row["accounting_log_disabled"] is disable_error_logs for row in rows)
    assert all(("status" in row) is not disable_error_logs for row in rows)
    assert all(("metadata" in row) is not disable_error_logs for row in rows)


@pytest.mark.parametrize(
    ("request_fields", "expected"),
    [
        pytest.param(
            {"cost": 1.0, "discount": 0.25, "customer_margin": None},
            finalize_cost(1.0, {"total_cost": 1.0}, 0.25),
            id="search_free_tier",
        ),
        pytest.param(
            {"cost": 1.0, "discount": None, "customer_margin": 0.1},
            apply_customer_cost_margin(finalize_cost(1.0, {"total_cost": 1.0}), 0.1),
            id="customer_surcharge",
        ),
        pytest.param(
            {"cost": 1.0, "discount": 0.25, "customer_margin": 0.1},
            apply_customer_cost_margin(finalize_cost(1.0, {"total_cost": 1.0}, 0.25), 0.1),
            id="both",
        ),
    ],
)
def test_the_charged_cost_reaches_the_live_logging_payload(
    request_fields: Mapping[str, object], expected: FinalizedCost
) -> None:
    payload: Final[dict[str, object]] = {"response_cost": 1.0, "cost_breakdown": {"total_cost": 1.0}}
    kwargs: Final[dict[str, object]] = {"response_cost": 1.0, "standard_logging_object": payload}
    token: Final = _SPEND_LOG_CALL.set(SpendLogCall(document={}, rows=(kwargs,), response=None))
    try:
        SpendLogHost().logged_cost(request_fields)
    finally:
        _SPEND_LOG_CALL.reset(token)

    assert payload["response_cost"] == pytest.approx(expected.amount)
    assert payload["cost_breakdown"] == expected.logging_breakdown()
    assert kwargs == {"response_cost": 1.0, "standard_logging_object": payload}


@pytest.mark.asyncio
@pytest.mark.parametrize("accounting_payload", [{"spend": 0.5}, None])
async def test_a_claimed_cost_row_reaches_the_post_write_steps_with_its_accounting_payload(
    monkeypatch: pytest.MonkeyPatch, accounting_payload: Mapping[str, object] | None
) -> None:
    after_write: Final = AfterWrite([])
    monkeypatch.setattr(ps.proxy_logging_obj.db_spend_update_writer, "record_after_ledger_row", after_write.record)
    token: Final = _SPEND_LOG_CALL.set(SpendLogCall(document={}, rows=({},), response=None))
    try:
        SpendLogHost().after_write(
            {
                "position": 0,
                "row": {"request_id": "video_abc_video_cost"},
                "claimed": accounting_payload is not None,
                **({} if accounting_payload is None else {"accounting_payload": accounting_payload}),
            }
        )
    finally:
        _SPEND_LOG_CALL.reset(token)
    for _ in range(100):
        if after_write.calls:
            break
        await asyncio.sleep(0.01)

    payload: Final = after_write.calls[0]["payload"]
    assert isinstance(payload, Mapping)
    assert payload.get("accounting_payload") == accounting_payload
    assert ("accounting_payload" in payload) is (accounting_payload is not None)


@pytest.mark.asyncio
async def test_a_row_that_failed_to_build_skips_the_accounting_windows_but_keeps_the_customer_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window_lookups: Final[list[object]] = []

    async def windows(*args: object, **kwargs: object) -> Sequence[object]:
        window_lookups.append(args)
        return ()

    async def customer_counters(*args: object, **kwargs: object) -> Sequence[BudgetCounter]:
        return (BudgetCounter("spend:customer:c1", 10.0, 0.0, "customer", "c1"),)

    monkeypatch.setattr(ps, "accounting_window_keys", windows)
    monkeypatch.setattr(ps, "customer_accounting_counters", customer_counters)
    token: Final = _SPEND_LOG_CALL.set(SpendLogCall(document={}, rows=({},), response=None))
    try:
        accounting: Final = await SpendLogHost().accounting_context(
            {"position": 0, "built": None, "token": "hashed-key", "charges": True}
        )
    finally:
        _SPEND_LOG_CALL.reset(token)

    assert window_lookups == []
    assert accounting["windows"] == []
    assert [window["counter_key"] for window in accounting["customer_windows"]] == ["spend:customer:c1"]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType, reportGeneralTypeIssues]  # the host reply is a JSON object


@pytest.mark.asyncio
async def test_a_responses_turn_replayed_from_the_database_wakes_the_monitor(proxy: Proxy) -> None:
    flush_requested: Final = asyncio.Event()
    proxy.client.spend_log_flush_requested = flush_requested

    _ = await litellm.aresponses(model="gpt-4o", input="hi", mock_response="hello there", metadata=dict(_KEYED))

    rows: Final = await proxy.rows()
    await asyncio.wait_for(flush_requested.wait(), timeout=2)

    assert len(rows) == 1
    assert flush_requested.is_set()


@pytest.mark.asyncio
async def test_the_proxy_registers_the_rust_callback_only_over_a_rust_writer(proxy: Proxy) -> None:
    assert rust_spend_loggers(proxy.client) is not None
    assert rust_spend_loggers(FakePrismaClient()) is None


async def _settled_row(proxy: Proxy, metadata: Mapping[str, object]) -> Mapping[str, object]:
    await _complete(metadata)
    rows: Final = await proxy.rows()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("margin", [0.1, 1])
async def test_a_customer_success_is_charged_the_surcharge_main_applies(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch, margin: float
) -> None:
    monkeypatch.setattr(litellm, "cost_margin_config", {"customer": margin})
    unsurcharged: Final = await _settled_row(proxy, _KEYED)

    surcharged: Final = await _settled_row(proxy, {**_KEYED, "user_api_key_customer_org_id": "customer-1"})

    base: Final = unsurcharged["spend"]
    assert isinstance(base, float) and base > 0
    expected: Final = apply_customer_cost_margin(
        finalize_cost(base, _metadata(unsurcharged).get("cost_breakdown")),  # pyright: ignore[reportArgumentType]  # the stored breakdown dict
        float(margin),
    )
    assert surcharged["spend"] == expected.amount
    assert _metadata(surcharged)["cost_breakdown"] == expected.logging_breakdown()
    accounting: Final = surcharged["accounting_payload"]
    assert isinstance(accounting, dict)
    assert accounting["spend"] == expected.amount
    assert accounting["metadata"]["accounting_customer_margin_spend"] == base * margin


class LoggedCosts(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.costs: Final[list[object]] = []

    async def async_log_success_event(
        self, kwargs: dict, response_obj: object, start_time: object, end_time: object
    ) -> None:  # pyright: ignore[reportMissingTypeArgument, reportIncompatibleMethodOverride]  # the legacy callback signature
        self.costs.append(kwargs["standard_logging_object"]["response_cost"])


@pytest.mark.asyncio
async def test_a_later_callback_logs_the_surcharged_cost(proxy: Proxy, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cost_margin_config", {"customer": 0.5})
    later: Final = LoggedCosts()
    monkeypatch.setattr(litellm, "callbacks", [proxy.logger, later])

    row: Final = await _settled_row(proxy, {**_KEYED, "user_api_key_customer_org_id": "customer-1"})

    for _ in range(300):
        if later.costs:
            break
        await asyncio.sleep(0.01)
    assert later.costs == [row["spend"]]


@pytest.mark.asyncio
async def test_a_billed_customer_failure_row_is_charged_the_surcharge_main_applies(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "cost_margin_config", {"customer": 0.5})

    await bill_failure_in_rust(
        _auth(),
        ValueError("the stream broke"),
        {
            "model": "gpt-4o",
            "metadata": {},
            "litellm_metadata": {"user_api_key_customer_org_id": "customer-1"},
            "response_cost": 2.0,
            "combined_usage_object": litellm.Usage(),
        },
    )

    rows: Final = await proxy.rows()
    assert len(rows) == 1
    expected: Final = snapshotless_cost_with_customer_margin(2.0, 0.5)
    assert rows[0]["spend"] == expected.amount == 3.0
    assert _metadata(rows[0])["cost_breakdown"] == expected.logging_breakdown()


def _attempt(outcome: Outcome, metadata: Mapping[str, object]) -> HookArgs:
    started: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return HookArgs(
        request={"model": "gpt-4o", "call_type": "acompletion", "litellm_params": {"metadata": dict(metadata)}},
        result=None,
        start_time=started,
        end_time=started,
        outcome=outcome,
    )


@pytest.mark.parametrize("outcome", [Outcome.failure, Outcome.fallback_failure])
def test_a_failed_attempt_gets_no_spend_log_input(outcome: Outcome) -> None:
    assert spend_log_success_input(_attempt(outcome, _KEYED)) is None


@pytest.mark.parametrize("outcome", [Outcome.success, Outcome.fallback_success, None])
def test_a_charged_attempt_carries_each_window_reset_as_the_counter_step_reads_it(outcome: Outcome | None) -> None:
    reset: Final = datetime(2026, 2, 1, tzinfo=timezone.utc)
    reservation: Final = {
        "reserved_cost": 1.0,
        "entries": [{"counter_key": "spend:key:hashed-key", "window_reset_at": reset}, {"counter_key": None}],
    }

    call: Final = spend_log_success_input(_attempt(outcome, {**_KEYED, "user_api_key_budget_reservation": reservation}))

    assert call is not None
    assert call.document["window_resets"] == [reset.isoformat(), None]
    assert call.document["window_resets"] == list(reservation_inputs(reservation)[1])


@pytest.mark.parametrize(
    ("configured", "sent"),
    [
        pytest.param(None, None, id="unset"),
        pytest.param({"exclude": ["usage_object"]}, {"include": None, "exclude": ["usage_object"]}, id="exclude"),
    ],
)
def test_each_attempt_carries_the_current_metadata_retention(
    monkeypatch: pytest.MonkeyPatch, configured: Mapping[str, object] | None, sent: Mapping[str, object] | None
) -> None:
    monkeypatch.setattr(ps, "general_settings", {"spend_logs_metadata_fields": configured})

    call: Final = spend_log_success_input(_attempt(Outcome.success, _KEYED))

    assert call is not None
    assert _SETTINGS.validate_python(call.document["settings"])["spend_logs_metadata_fields"] == sent
