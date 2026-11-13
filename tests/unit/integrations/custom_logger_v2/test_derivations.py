import json
from collections.abc import Generator, Mapping
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Annotated, Final, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, TypeAdapter

from litellm.integrations.custom_logger_v2 import derivations
from litellm.integrations.custom_logger_v2.derivations import RECORD_TYPES
from litellm.integrations.custom_logger_v2.slots import HookArgs
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.integrations.custom_logger_v2 import (
    AttemptTiming,
    BudgetPlan,
    BudgetPlanCounter,
    BudgetReservation,
    KeyIdentity,
    KeyLimits,
    Moment,
    Outcome,
    Record,
    Usage,
    UsageRecord,
    coerce_record,
    reset_coercion_reports_for_tests,
)

RECORDS_JSON: Final = Path(__file__).parents[4] / "litellm/rust/crates/callbacks/records.json"
_SCALARS: Final[Mapping[object, object]] = {str: "text", bool: True, int: 7, float: 1.5}
_WRONG: Final[Mapping[object, object]] = {str: 7, bool: "maybe", int: "seven", float: "one and a half"}


@pytest.fixture(autouse=True)
def fresh_reports() -> Generator[None]:
    reset_coercion_reports_for_tests()
    yield
    reset_coercion_reports_for_tests()


def _bare(annotation: object) -> object:
    origin: Final = get_origin(annotation)
    args: Final = get_args(annotation)
    if origin is Annotated:
        return _bare(args[0])
    if origin in (Union, UnionType):
        return _bare(next(arg for arg in args if arg is not type(None)))
    return annotation


def _sample(annotation: object) -> object:
    bare: Final = _bare(annotation)
    origin: Final = get_origin(bare)
    args: Final = get_args(bare)
    if bare in _SCALARS:
        return _SCALARS[bare]
    if origin is tuple:
        return [_sample(args[0])]
    if origin is Mapping:
        return {"x-ratelimit-key-remaining-requests": _sample(args[1])}
    if isinstance(bare, type) and issubclass(bare, Enum):
        return list(bare)[-1].value
    if isinstance(bare, type) and issubclass(bare, BaseModel):
        return {name: _sample(member.annotation) for name, member in bare.model_fields.items()}
    raise TypeError(f"no sample for {annotation!r}")


def _record_samples() -> Mapping[str, object]:
    return {
        field.value: TypeAdapter(annotation).dump_python(
            TypeAdapter(annotation).validate_python(_sample(annotation)), mode="json"
        )
        for field, annotation in sorted(RECORD_TYPES.items(), key=lambda item: item[0].bit)
    }


def _records() -> tuple[type[Record], ...]:
    return tuple(record for record in RECORD_TYPES.values() if isinstance(record, type) and issubclass(record, Record))


def test_every_record_sample_matches_the_file_the_rust_serde_types_decode() -> None:
    rendered: Final = json.dumps(_record_samples(), indent=2) + "\n"

    assert rendered == RECORDS_JSON.read_text(), f"regenerate {RECORDS_JSON} from _record_samples()"


@pytest.mark.parametrize("record", _records(), ids=lambda record: record.__name__)
def test_a_member_that_cannot_be_coerced_is_left_absent_and_logged_once(
    record: type[Record], caplog: pytest.LogCaptureFixture
) -> None:
    name, member = next(iter(record.model_fields.items()))
    good: Final = {other: _sample(info.annotation) for other, info in record.model_fields.items() if other != name}
    wrong: Final = _WRONG.get(_bare(member.annotation), 7)

    with caplog.at_level("WARNING", logger="LiteLLM"):
        first: Final = coerce_record(record, {**good, name: wrong})
        second: Final = coerce_record(record, {**good, name: wrong})

    assert first == second == record.model_validate(good)
    assert getattr(first, name) is None
    assert [entry.getMessage().split(":")[1].strip() for entry in caplog.records] == [
        f"{record.__name__}.{name} could not be coerced, so it is left absent"
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, 1), (" 42 ", 42), (2**63 - 1, 2**63 - 1), (2**63, None), (4.0, 4), (4.5, None), ("1e3", None)],
)
def test_an_integer_member_takes_what_pydantic_coerces_and_drops_what_does_not_fit_an_i64(
    raw: object, expected: int | None
) -> None:
    assert coerce_record(UsageRecord, {"prompt_tokens": raw}).prompt_tokens == expected


@pytest.mark.parametrize(
    ("raw", "expected"), [(" 0.5 ", 0.5), (1, 1.0), (float("nan"), None), (float("inf"), None), (10**400, None)]
)
def test_a_float_member_takes_what_pydantic_coerces_and_drops_what_json_cannot_carry(
    raw: object, expected: float | None
) -> None:
    assert coerce_record(KeyLimits, {"budget_throttle_pct": raw}).budget_throttle_pct == expected


def test_a_budget_plan_status_code_outside_u16_is_left_absent() -> None:
    assert coerce_record(BudgetPlan, {"status_code": 70000, "fail_closed": True}) == BudgetPlan(fail_closed=True)


def test_a_moment_keeps_the_datetime_fields_and_its_offset() -> None:
    start: Final = datetime(2026, 11, 13, 9, 30, 15, 250, tzinfo=timezone(timedelta(hours=-5)))

    timing: Final = coerce_record(AttemptTiming, {"start": start, "end": "yesterday"})

    assert timing == AttemptTiming(
        start=Moment(
            year=2026, month=11, day=13, hour=9, minute=30, second=15, microsecond=250, utc_offset_seconds=-18000
        )
    )


def _auth(**members: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-real", user_id="real-user", team_id="real-team", **members)


def test_key_identity_comes_from_the_auth_object_and_never_from_the_request_body() -> None:
    auth: Final = _auth(team_metadata={"priority": "high"}, metadata={"priority": 3})
    auth.invoked_agent_id = "agent-1"
    forged: Final = {
        "user_api_key_dict": {"user_id": "forged"},
        "metadata": {"user_api_key_user_id": "forged", "user_api_key_team_id": "forged"},
        "litellm_metadata": {"user_api_key_hash": "forged"},
    }

    identity: Final = derivations.key_identity(HookArgs(auth=auth, request=forged))

    assert identity == KeyIdentity(
        key_hash=auth.token,
        api_key=auth.api_key,
        user_id="real-user",
        team_id="real-team",
        invoked_agent_id="agent-1",
        team_priority="high",
    )
    assert derivations.key_identity(HookArgs(request=forged)) is None


def test_key_limits_read_the_attributes_model_dump_excludes() -> None:
    auth: Final = _auth(rpm_limit=10, metadata={"default_estimated_output_tokens": 128})
    auth.budget_throttle_pct = 0.8

    limits: Final = derivations.key_limits(HookArgs(auth=auth, request={"model": "gpt"}))

    assert "budget_throttle_pct" not in auth.model_dump()
    assert limits is not None
    assert limits.budget_throttle_pct == 0.8
    assert limits.rpm_limit == 10
    assert limits.estimated_output_tokens is not None
    assert limits.estimated_output_tokens.key_default == 128


def test_a_budget_plan_is_claimed_once_so_a_rerun_pre_call_sees_none() -> None:
    plan: Final = BudgetPlan(counters=(BudgetPlanCounter(counter_key="k", max_budget=5.0),), reservation_cost=0.25)
    auth: Final = _auth(budget_plan=plan)

    first: Final = derivations.claim_budget_plan(HookArgs(auth=auth))
    rerun: Final = derivations.claim_budget_plan(HookArgs(auth=auth))

    assert first == plan
    assert rerun is None
    assert auth.budget_plan is None


def test_a_budget_reservation_write_updates_the_shared_dict_in_place() -> None:
    reservation: Final[dict[str, object]] = {"reserved_cost": 0.5, "finalized": False}
    auth: Final = _auth()
    auth.budget_reservation = reservation
    args: Final = HookArgs(auth=auth)

    derivations.write_budget_reservation(args, {"finalized": True, "input_tokens": 12, "input_cost": None})

    assert auth.budget_reservation is reservation
    assert reservation == {"reserved_cost": 0.5, "finalized": True, "input_tokens": 12}
    assert derivations.budget_reservation(args) == BudgetReservation(reserved_cost=0.5, finalized=True, input_tokens=12)


def test_usage_keeps_each_source_apart() -> None:
    request: Final = {
        "combined_usage_object": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
        "standard_logging_object": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "metadata": {}},
    }
    result: Final = {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}

    usage: Final = derivations.usage(HookArgs(request=request, result=result))

    assert usage == Usage(
        response=UsageRecord(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        combined=UsageRecord(prompt_tokens=9, completion_tokens=1, total_tokens=10),
        logged=UsageRecord(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )


def test_the_payload_copy_is_shallow_so_a_later_top_level_write_does_not_reach_it() -> None:
    nested: Final[dict[str, object]] = {"a": 1}
    payload: Final[dict[str, object]] = {"id": "log-1", "metadata": nested}

    copied: Final = derivations.payload_copy(HookArgs(request={"standard_logging_object": payload}))
    payload["id"] = "changed"

    assert copied == {"id": "log-1", "metadata": nested}
    assert copied is not None
    assert copied["metadata"] is nested


def test_the_outcome_is_the_one_the_hook_was_called_with() -> None:
    assert derivations.outcome(HookArgs(outcome=Outcome.fallback_failure)) is Outcome.fallback_failure
    assert Outcome.fallback_success.value == "fallback_success"
