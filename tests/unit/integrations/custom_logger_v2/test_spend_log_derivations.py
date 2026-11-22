import asyncio
import copy
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, TypedDict, cast

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger_v2 import CallContext, CustomLoggerV2, Field, Hook, Patch, on
from litellm.integrations.custom_logger_v2 import derivations
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import BilledFailedAttempt, ProxyLogging
from litellm.types.integrations.custom_logger_v2 import BudgetReservation, Cost, SpendLogId
from litellm.types.utils import Usage


class _Goldens(TypedDict):
    recorded_from: str
    success: list[dict[str, object]]
    failure: list[dict[str, object]]


_GOLDENS: Final = TypeAdapter(_Goldens).validate_json((Path(__file__).parent / "spend_log_goldens.json").read_text())
_SUCCESS: Final = _GOLDENS["success"]
_FAILURE: Final = _GOLDENS["failure"]
_COUNTS: Final = TypeAdapter(Mapping[str, int])
_BASE: Final[Mapping[str, object]] = {
    "model": "gpt-4.1",
    "messages": [{"role": "user", "content": "Hello"}],
    "call_type": "acompletion",
}
_RESERVATION: Final[Mapping[str, object]] = {
    "reserved_cost": 0.75,
    "entries": [
        {
            "counter_key": "spend:key:k",
            "entity_type": "key",
            "entity_id": "k",
            "reserved_cost": 0.75,
            "window_reset_at": "2026-11-19T00:00:00+00:00",
        }
    ],
    "finalized": False,
}


def _named(scenarios: Sequence[dict[str, object]]) -> list[object]:
    return [pytest.param(scenario, id=str(scenario["name"])) for scenario in scenarios]


def _usage(raw: object) -> Usage:
    counts: Final = _COUNTS.validate_python(raw)
    return Usage(
        prompt_tokens=counts["prompt_tokens"],
        completion_tokens=counts["completion_tokens"],
        total_tokens=counts["total_tokens"],
    )


def _attempt(raw: Mapping[str, object]) -> BilledFailedAttempt:
    return BilledFailedAttempt(
        attempted_retries=cast(int, raw["attempted_retries"]),
        deployment_id=cast("str | None", raw["deployment_id"]),
        usage=_usage(raw["usage"]),
        response_cost=cast(float, raw["response_cost"]),
    )


def _request(raw: object) -> dict[str, object]:
    request: Final = {**_BASE, **copy.deepcopy(cast("dict[str, object]", raw))}
    if "combined_usage_object" in request:
        request["combined_usage_object"] = _usage(request["combined_usage_object"])
    if "billed_failed_attempts" in request:
        attempts: Final = cast("list[Mapping[str, object]]", request["billed_failed_attempts"])
        request["billed_failed_attempts"] = tuple(map(_attempt, attempts))
    return request


@pytest.mark.parametrize("scenario", _named(_SUCCESS))
def test_the_spend_log_id_is_the_recorded_request_id(scenario: dict[str, object]) -> None:
    derived: Final = derivations.spend_log_id(
        HookArgs(request=_request(scenario["request"]), result=scenario["response"])
    )

    assert derived == SpendLogId(id=cast(str, scenario["id"]))


@pytest.mark.parametrize("scenario", _named(_FAILURE))
def test_failure_ids_costs_and_billed_attempts_are_the_recorded_rows(scenario: dict[str, object]) -> None:
    args: Final = HookArgs(auth=UserAPIKeyAuth(api_key="hashed-key"), request=_request(scenario["request"]))
    rows: Final = cast("list[Mapping[str, object]]", scenario["rows"])

    final_id: Final = derivations.failure_spend_log_id(args)
    cost: Final = derivations.failure_cost(args)
    attempts: Final = derivations.billed_retry_attempts(args) or ()

    assert final_id is not None and cost is not None
    assert [str(final_id.id), *(attempt.spend_log_id.id for attempt in attempts if attempt.spend_log_id)] == [
        row["id"] for row in rows
    ]
    assert [cost.failure_row_response_cost, *(attempt.response_cost for attempt in attempts)] == [
        row["response_cost"] for row in rows
    ]
    assert [(attempt.litellm_call_id, attempt.parent_request_id) for attempt in attempts] == [
        (row["litellm_call_id"], row["parent_request_id"]) for row in rows[1:]
    ]
    assert [attempt.position for attempt in attempts] == list(range(len(rows) - 1))


def test_billed_attempts_carry_their_retry_count_deployment_and_usage() -> None:
    (scenario,) = (scenario for scenario in _FAILURE if scenario["name"] == "call_id_rows")

    attempts: Final = derivations.billed_retry_attempts(HookArgs(request=_request(scenario["request"])))

    assert attempts is not None
    assert [(attempt.attempted_retries, attempt.deployment_id) for attempt in attempts] == [(0, "dep-a"), (1, None)]
    assert [attempt.usage.prompt_tokens for attempt in attempts if attempt.usage] == [24, 24]


def test_a_failure_without_billed_attempts_has_none() -> None:
    assert derivations.billed_retry_attempts(HookArgs(request={"litellm_call_id": "call-2"})) is None


@pytest.mark.parametrize(
    ("payload", "finalized", "debug_info"),
    [
        ({"cost_finalized": True, "response_cost_failure_debug_info": None}, True, None),
        ({"response_cost_failure_debug_info": {"error_str": "no price", "traceback_str": "tb"}}, False, "no price"),
        ({"cost_finalized": 1}, False, None),
    ],
    ids=["finalized", "debug_info", "truthy_is_not_finalized"],
)
def test_cost_carries_the_finalized_flag_and_the_cost_failure_reason(
    payload: Mapping[str, object], finalized: bool, debug_info: str | None
) -> None:
    cost: Final = derivations.cost(HookArgs(request={"standard_logging_object": payload}))

    assert cost == Cost(cost_finalized=finalized, response_cost_failure_debug_info=debug_info)


class _SpendReader(CustomLoggerV2):
    def __init__(self) -> None:
        self.seen: list[tuple[object, object]] = []  # mutable-ok: what attempt_ended hands the port
        self.done = threading.Event()

    @on(Hook.attempt_ended, reads={Field.spend_log_id, Field.budget_reservation}, awaited=True)
    async def attempt_ended(self, view: object, ctx: CallContext[object]) -> None:
        self.seen.append((getattr(view, "spend_log_id", None), getattr(view, "budget_reservation", None)))
        self.done.set()


class _Settler(CustomLoggerV2):
    def __init__(self) -> None:
        self.settled: list[object] = []  # mutable-ok: the reservations the settle hooks were handed
        self.failure_reads: tuple[object, object] = (None, None)

    def _settle(self, view: object) -> Patch | None:
        reservation: Final = getattr(view, "budget_reservation", None)
        self.settled.append(reservation)
        if not isinstance(reservation, BudgetReservation):
            return None
        entries: Final = tuple(
            entry.model_copy(update={"applied_adjustment": 0.5}) if entry is not None else None
            for entry in reservation.entries or ()
        )
        return Patch().set(
            Field.budget_reservation, reservation.model_copy(update={"finalized": True, "entries": entries})
        )

    @on(Hook.attempt_ended, reads={Field.budget_reservation}, writes={Field.budget_reservation})
    async def attempt_ended(self, view: object, ctx: CallContext[object]) -> Patch | None:
        return self._settle(view)

    @on(
        Hook.post_call_failure,
        reads={Field.budget_reservation, Field.spend_log_id, Field.cost},
        writes={Field.budget_reservation},
    )
    async def post_call_failure(self, view: object, ctx: CallContext[object]) -> Patch | None:
        self.failure_reads = (getattr(view, "spend_log_id", None), getattr(view, "cost", None))
        return self._settle(view)


def _only(callbacks: Sequence[object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", list(callbacks))
    for name in ("input_callback", "success_callback", "failure_callback", "_async_success_callback"):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])


async def _attempt_ended(monkeypatch: pytest.MonkeyPatch, *extra: CustomLoggerV2, **call: object) -> tuple[object, ...]:
    reader: Final = _SpendReader()
    _only((*extra, reader), monkeypatch)
    response: Final = await litellm.acompletion(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], mock_response="ok", **call
    )
    ids: Final[set[object]] = set()
    if isinstance(response, litellm.CustomStreamWrapper):
        async for chunk in response:  # pyright: ignore[reportUnknownVariableType]  # the stream yields untyped chunks
            ids.add(getattr(chunk, "id", None))  # pyright: ignore[reportUnknownArgumentType]  # an untyped chunk
    else:
        ids.add(getattr(response, "id", None))
    assert await asyncio.to_thread(reader.done.wait, 30)
    (spend_log_id, reservation) = reader.seen[0]
    return ids, spend_log_id, reservation


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
async def test_attempt_ended_names_the_row_after_the_response_it_logs(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    ids, spend_log_id, _ = await _attempt_ended(monkeypatch, stream=stream)

    assert isinstance(spend_log_id, SpendLogId)
    assert ids == {spend_log_id.id}


@pytest.mark.parametrize(
    "metadata",
    [
        {"user_api_key_budget_reservation": "stamped"},
        {"user_api_key_auth": {"budget_reservation": "stamped"}},
    ],
    ids=["stamped", "auth_fallback"],
)
async def test_attempt_ended_reads_and_settles_the_logged_reservation_in_place(
    monkeypatch: pytest.MonkeyPatch, metadata: Mapping[str, object]
) -> None:
    held: Final = copy.deepcopy(dict(_RESERVATION))
    placed: Final = {
        key: held if value == "stamped" else {"budget_reservation": held} for key, value in metadata.items()
    }
    settler: Final = _Settler()

    _, _, reservation = await _attempt_ended(monkeypatch, settler, metadata=placed)

    assert isinstance(reservation, BudgetReservation) and reservation.reserved_cost == 0.75
    assert held["finalized"] is True
    (entry,) = cast("list[dict[str, object]]", held["entries"])
    assert entry["applied_adjustment"] == 0.5
    assert entry["window_reset_at"] == "2026-11-19T00:00:00+00:00"


def test_the_logged_reservation_prefers_the_litellm_metadata_bucket_and_keeps_the_key_stamp() -> None:
    stamped: Final = copy.deepcopy(dict(_RESERVATION))
    other: Final = {**copy.deepcopy(dict(_RESERVATION)), "reserved_cost": 2.0}
    args: Final = HookArgs(
        request={
            "litellm_params": {
                "metadata": {"user_api_key_budget_reservation": stamped},
                "litellm_metadata": {"user_api_key_auth": {"budget_reservation": other}},
            }
        }
    )

    reservation: Final = derivations.logged_budget_reservation(args)

    assert isinstance(reservation, BudgetReservation) and reservation.reserved_cost == 0.75


def test_a_request_without_a_reservation_reads_none() -> None:
    assert derivations.logged_budget_reservation(HookArgs(request={"litellm_params": {"metadata": {}}})) is None


async def test_post_call_failure_reads_the_spend_facts_and_settles_the_auth_reservation_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settler: Final = _Settler()
    _only((settler,), monkeypatch)
    auth: Final = UserAPIKeyAuth(api_key="hashed-key", budget_reservation=copy.deepcopy(dict(_RESERVATION)))
    held: Final = auth.budget_reservation
    assert held is not None
    request: Final[dict[str, object]] = {**_BASE, "litellm_call_id": "call-7", "metadata": {}}

    _ = await ProxyLogging(user_api_key_cache=UserApiKeyCache()).post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # the hook takes the untyped request dict
        request_data=request, original_exception=Exception("upstream broke"), user_api_key_dict=auth
    )

    (reservation,) = settler.settled
    spend_log_id, cost = settler.failure_reads
    assert isinstance(reservation, BudgetReservation) and reservation.reserved_cost == 0.75
    assert spend_log_id == SpendLogId(id="call-7")
    assert isinstance(cost, Cost) and cost.failure_row_response_cost == 0.0
    assert held["finalized"] is True
    (entry,) = cast("list[dict[str, object]]", held["entries"])
    assert entry["applied_adjustment"] == 0.5


def test_a_settle_write_without_a_held_reservation_changes_nothing() -> None:
    auth: Final = UserAPIKeyAuth(api_key="hashed-key")

    derivations.write_settled_budget_reservation(HookArgs(auth=auth), BudgetReservation(finalized=True))

    assert auth.budget_reservation is None
