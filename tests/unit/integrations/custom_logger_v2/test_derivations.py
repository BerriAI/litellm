import json
import math
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
    Cost,
    Deployment,
    EstimatedOutputTokens,
    GuardrailEntry,
    KeyIdentity,
    KeyLimits,
    LoggedKeyIdentity,
    MetadataKind,
    ModelInfo,
    ModelPreference,
    Moment,
    Outcome,
    RateLimitState,
    Record,
    Scalar,
    ScalarItem,
    Usage,
    UsageRecord,
    ZeroCost,
    coerce_record,
    reset_coercion_reports_for_tests,
)

RECORDS_JSON: Final = Path(__file__).parents[4] / "litellm/rust/crates/callbacks/records.json"
_SCALARS: Final[Mapping[object, object]] = {
    Scalar: 42,
    ScalarItem: " 7 ",
    str: "text",
    bool: True,
    int: 7,
    float: 1.5,
    tuple[str, ...]: ["member"],
}
_WRONG: Final[Mapping[object, object]] = {Scalar: ["a1"], str: 7, bool: "maybe", int: "seven", float: "one and a half"}
_MOMENT: Final = datetime(2026, 11, 13, 9, 30, 15, 250, tzinfo=timezone(timedelta(hours=-5)))


@pytest.fixture(autouse=True)
def fresh_reports() -> Generator[None]:
    reset_coercion_reports_for_tests()
    yield
    reset_coercion_reports_for_tests()


def _bare(annotation: object) -> object:
    origin: Final = get_origin(annotation)
    args: Final = get_args(annotation)
    if annotation in _SCALARS:
        return annotation
    if origin is Annotated and args[0] is Moment:
        return datetime
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
    if bare is datetime:
        return _MOMENT
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


def _first_member(record: type[Record]) -> str:
    return next(name for name in record.model_fields if name != "rejected")


@pytest.mark.parametrize("record", _records(), ids=lambda record: record.__name__)
def test_a_member_no_python_consumer_can_use_is_named_in_rejected_and_logged_once(
    record: type[Record], caplog: pytest.LogCaptureFixture
) -> None:
    name: Final = _first_member(record)
    good: Final = {
        other: _sample(info.annotation)
        for other, info in record.model_fields.items()
        if other not in (name, "rejected")
    }
    wrong: Final = _WRONG.get(_bare(record.model_fields[name].annotation), ["a1"])

    with caplog.at_level("WARNING", logger="LiteLLM"):
        first: Final = coerce_record(record, {**good, name: wrong})
        second: Final = coerce_record(record, {**good, name: wrong})

    assert first == second == record.model_validate({**good, "rejected": (name,)})
    assert getattr(first, name) is None
    assert [entry.getMessage().split(":")[1].strip() for entry in caplog.records] == [
        f"{record.__name__}.{name} holds a value the contract cannot carry, so it is named in rejected"
    ]


@pytest.mark.parametrize(
    "raw",
    [True, False, 0, " 42 ", "ten", 2**63, -(2**100), 4.0, 4.5, "1e3", float("inf"), "", "None"],
    ids=repr,
)
def test_a_scalar_member_keeps_the_value_and_type_python_reads(raw: object, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="LiteLLM"):
        record: Final = coerce_record(UsageRecord, {"prompt_tokens": raw})

    assert record.prompt_tokens == raw
    assert type(record.prompt_tokens) is type(raw)
    assert record.rejected is None
    assert caplog.records == []


def test_nan_is_carried_as_the_float_python_reads() -> None:
    cost: Final = coerce_record(Cost, {"response_cost": float("nan")})

    assert isinstance(cost.response_cost, float)
    assert math.isnan(cost.response_cost)


@pytest.mark.parametrize("raw", [2**127, -(2**127) - 1, ["a1"], {"a": 1}, b"7"], ids=repr)
def test_a_scalar_member_rejects_what_is_not_a_scalar_or_overflows_i128(raw: object) -> None:
    assert coerce_record(UsageRecord, {"prompt_tokens": raw}) == UsageRecord(rejected=("prompt_tokens",))


def test_a_none_member_is_absent_like_a_missing_one_and_never_rejected() -> None:
    assert coerce_record(KeyLimits, {"rpm_limit": None}) == KeyLimits() == coerce_record(KeyLimits, {})


def test_a_budget_plan_status_code_outside_u16_is_named_in_rejected() -> None:
    assert coerce_record(BudgetPlan, {"status_code": 70000, "fail_closed": True}) == BudgetPlan(
        fail_closed=True, rejected=("status_code",)
    )


def test_a_moment_keeps_the_datetime_fields_and_its_offset_and_rejects_what_is_not_a_datetime() -> None:
    timing: Final = coerce_record(
        AttemptTiming, {"start": _MOMENT, "end": "yesterday", "api_call_start": {"year": 2026}}
    )

    assert timing == AttemptTiming(
        start=Moment(
            year=2026, month=11, day=13, hour=9, minute=30, second=15, microsecond=250, utc_offset_seconds=-18000
        ),
        rejected=("end", "api_call_start"),
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
        key_priority=3,
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


@pytest.mark.parametrize(
    ("member", "raw"),
    [
        ("end_user_id", 123),
        ("end_user_id", 1.5),
        ("end_user_id", True),
        ("project_id", 9),
        ("team_id", 10**20),
        ("user_id", ""),
    ],
    ids=repr,
)
def test_key_identity_keeps_a_non_string_id_the_limiter_formats_with_python_str(member: str, raw: object) -> None:
    auth: Final = UserAPIKeyAuth(api_key="sk-real")
    setattr(auth, member, raw)

    identity: Final = derivations.key_identity(HookArgs(auth=auth, request={"model": "gpt"}))

    assert identity is not None
    assert identity.model_dump_json(exclude_none=True, exclude={"key_hash", "api_key"}) == json.dumps(
        {member: raw}, separators=(",", ":")
    )


@pytest.mark.parametrize("priority", [1, 2.5, True, "high"], ids=repr)
def test_key_identity_keeps_numeric_priorities_capacity_compares_by_value(priority: object) -> None:
    identity: Final = derivations.key_identity(
        HookArgs(auth=_auth(team_metadata={"priority": priority}, metadata={"priority": priority}))
    )

    assert identity is not None
    assert (identity.team_priority, identity.key_priority) == (priority, priority)


def test_key_identity_names_an_unhashable_priority_and_a_list_agent_id_in_rejected() -> None:
    auth: Final = _auth(team_metadata={"priority": ["gold"]})
    auth.agent_id = ["a1"]  # pyright: ignore[reportAttributeAccessIssue]  # the shape a hand-built auth object can hold

    identity: Final = derivations.key_identity(HookArgs(auth=auth))

    assert identity is not None
    assert identity.rejected == ("agent_id", "team_priority")
    assert (identity.agent_id, identity.team_priority) == (None, None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([{"model": "gpt", "alternates": ["mini"]}, {"model": "other"}], (("gpt", ("mini",)), ("other", ()))),
        ([{"model": "gpt", "alternates": [7]}], None),
        ({"model": "gpt"}, None),
    ],
    ids=("parsed", "alternate-not-a-string", "not-a-list"),
)
def test_model_preferences_follow_the_capacity_parse_and_are_dropped_when_it_ignores_them(
    raw: object, expected: tuple[tuple[str, tuple[str, ...]], ...] | None
) -> None:
    identity: Final = derivations.key_identity(HookArgs(auth=_auth(team_metadata={"model_preferences": raw})))

    assert identity is not None
    assert identity.model_preferences == (
        None if expected is None else tuple(ModelPreference(model=model, alternates=alt) for model, alt in expected)
    )
    assert identity.rejected is None


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({"rpm_limit": "2"}, {"rpm_limit": "2"}),
        ({"tpm_limit": " 20 "}, {"tpm_limit": " 20 "}),
        ({"tpm_limit": 2.5}, {"tpm_limit": 2.5}),
        ({"max_parallel_requests": True}, {"max_parallel_requests": True}),
        ({"budget_throttle_pct": "0.5"}, {"budget_throttle_pct": "0.5"}),
        ({"rpm_limit": 0}, {"rpm_limit": 0}),
    ],
    ids=repr,
)
def test_key_limits_keep_the_raw_limit_python_converts_at_use(
    limits: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    auth: Final = _auth()
    for name, value in limits.items():
        setattr(auth, name, value)

    record: Final = derivations.key_limits(HookArgs(auth=auth, request={"model": "gpt"}))

    assert record is not None
    assert record.model_dump(exclude_none=True, exclude={"estimated_output_tokens"}) == expected


def test_a_limit_type_that_is_not_a_string_is_ignored_like_the_limiter_ignores_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="LiteLLM"):
        record: Final = derivations.key_limits(
            HookArgs(auth=_auth(metadata={"rpm_limit_type": "dynamic", "tpm_limit_type": 5}), request={})
        )

    assert record is not None
    assert (record.rpm_limit_type, record.tpm_limit_type, record.rejected) == ("dynamic", None, None)
    assert caplog.records == []


def test_estimated_output_tokens_use_the_limiter_parse_of_each_metadata_blob() -> None:
    auth: Final = _auth(
        metadata={
            "default_estimated_output_tokens_per_model": {"gpt": "64", "bad": -1},
            "default_estimated_output_tokens": "32",
        },
        team_metadata={"default_estimated_output_tokens_per_model": {"gpt": 16}, "default_estimated_output_tokens": 0},
    )

    with_model: Final = derivations.key_limits(HookArgs(auth=auth, request={"model": "gpt"}))
    without_model: Final = derivations.key_limits(HookArgs(auth=auth, request={"model": ["gpt"]}))

    assert with_model is not None
    assert with_model.estimated_output_tokens == EstimatedOutputTokens(key_default=32, team_per_model=16)
    assert without_model is not None
    assert without_model.estimated_output_tokens == EstimatedOutputTokens(key_default=32)


def _logging_request(metadata: object = None, **members: object) -> Mapping[str, object]:
    payload: Final = {"id": "chatcmpl-1"} if metadata is None else {"id": "chatcmpl-1", "metadata": metadata}
    return {"standard_logging_object": payload, **members}


_EVERY_STAMPED_KEY: Final = (
    "user_api_key_hash",
    "user_api_key_alias",
    "user_api_key_spend",
    "user_api_key_max_budget",
    "user_api_key_budget_reset_at",
    "user_api_key_user_id",
    "user_api_key_user_email",
    "user_api_key_user_spend",
    "user_api_key_user_max_budget",
    "user_api_key_team_id",
    "user_api_key_team_alias",
    "user_api_key_team_spend",
    "user_api_key_team_max_budget",
    "user_api_key_org_id",
    "user_api_key_org_alias",
    "user_api_key_project_id",
    "user_api_key_project_alias",
    "user_api_key_end_user_id",
    "user_api_key_request_route",
    "user_api_key_auth_metadata",
)


def test_logged_key_identity_comes_from_the_payload_metadata_the_loggers_read() -> None:
    metadata: Final = {
        **dict.fromkeys(_EVERY_STAMPED_KEY),
        "user_api_key_hash": "session-alias",
        "user_api_key_alias": "alias",
        "user_api_key_team_id": 10**20,
        "user_api_key_team_alias": "",
        "user_api_key_user_id": "u1",
        "user_api_key_org_id": "o1",
        "user_api_key_project_id": 9,
        "user_api_key_end_user_id": 123,
        "user_api_key_request_route": "/anthropic/v1/messages",
        "user_api_key_spend": 1.25,
        "user_api_key_auth_metadata": {"priority": 2},
        "agent_id": "a1",
    }

    identity: Final = derivations.logged_key_identity(
        HookArgs(
            request=_logging_request(
                metadata,
                user=1.5,
                litellm_params={"metadata": {"user_api_key_end_user_id": "from-params"}},
            ),
            logging_obj=None,
        )
    )

    assert identity == LoggedKeyIdentity(
        metadata_kind=MetadataKind.mapping,
        key_hash="session-alias",
        key_alias="alias",
        key_spend=1.25,
        team_id=10**20,
        team_alias="",
        user_id="u1",
        org_id="o1",
        project_id=9,
        end_user_id=123,
        request_route="/anthropic/v1/messages",
        auth_metadata_priority=2,
        agent_id="a1",
        request_user=1.5,
        cost_tracking_end_user_id="from-params",
    )


def test_logged_key_identity_never_reads_the_auth_object() -> None:
    auth: Final = _auth()

    identity: Final = derivations.logged_key_identity(
        HookArgs(
            auth=auth, request=_logging_request({**dict.fromkeys(_EVERY_STAMPED_KEY), "user_api_key_user_id": "logged"})
        )
    )

    assert identity == LoggedKeyIdentity(metadata_kind=MetadataKind.mapping, user_id="logged")


def test_a_stamped_key_missing_from_the_payload_metadata_differs_from_one_set_to_none() -> None:
    present: Final = {**dict.fromkeys(_EVERY_STAMPED_KEY), "user_api_key_user_id": None}
    missing: Final = {key: value for key, value in present.items() if key != "user_api_key_user_id"}

    with_none: Final = derivations.logged_key_identity(HookArgs(request=_logging_request(present)))
    without: Final = derivations.logged_key_identity(HookArgs(request=_logging_request(missing)))

    assert with_none == LoggedKeyIdentity(metadata_kind=MetadataKind.mapping)
    assert without == LoggedKeyIdentity(metadata_kind=MetadataKind.mapping, missing_keys=("user_api_key_user_id",))


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ({"id": "x"}, MetadataKind.missing),
        ({"id": "x", "metadata": None}, MetadataKind.null),
        ({"id": "x", "metadata": "text"}, MetadataKind.other),
        ({"id": "x", "metadata": {}}, MetadataKind.mapping),
    ],
    ids=("missing", "null", "other", "mapping"),
)
def test_logged_key_identity_says_how_the_payload_metadata_was_shaped(
    payload: Mapping[str, object], kind: MetadataKind
) -> None:
    identity: Final = derivations.logged_key_identity(HookArgs(request={"standard_logging_object": payload}))

    assert identity == LoggedKeyIdentity(
        metadata_kind=kind, missing_keys=_EVERY_STAMPED_KEY if kind is MetadataKind.mapping else None
    )


def test_logged_auth_metadata_that_is_not_a_mapping_is_named_so_capacity_can_refuse_settlement() -> None:
    identity: Final = derivations.logged_key_identity(
        HookArgs(request=_logging_request({**dict.fromkeys(_EVERY_STAMPED_KEY), "user_api_key_auth_metadata": ["p"]}))
    )

    assert identity == LoggedKeyIdentity(metadata_kind=MetadataKind.mapping, rejected=("auth_metadata",))


def test_the_cost_tracking_end_user_prefers_the_metadata_bucket_like_the_merge_python_applies() -> None:
    params: Final = {
        "litellm_metadata": {"user_api_key_end_user_id": "litellm-bucket", "other": 1},
        "metadata": {"user_api_key_end_user_id": "metadata-bucket"},
    }

    identity: Final = derivations.logged_key_identity(HookArgs(request=_logging_request({}, litellm_params=params)))

    assert identity is not None
    assert identity.cost_tracking_end_user_id == "metadata-bucket"
    assert params["litellm_metadata"] == {"user_api_key_end_user_id": "litellm-bucket", "other": 1}


def test_prometheus_parity_values_cross_as_python_holds_them() -> None:
    request: Final = {
        "standard_logging_object": {
            "prompt_tokens": "ten",
            "completion_tokens": float("inf"),
            "total_tokens": [10],
            "response_cost": 0.5,
            "zero_cost_diagnostic": {"other": 1},
            "hidden_params": {
                "litellm_overhead_time_ms": "slow",
                "additional_headers": {"x_ratelimit_remaining_requests": " 77 ", "x-ratelimit-key-remaining": True},
            },
            "metadata": {},
        },
        "litellm_params": {
            "metadata": {
                "model_group": "g",
                "queue_time_seconds": "slow",
                "litellm-key-remaining-requests-g": "250",
            }
        },
    }
    args: Final = HookArgs(request=request)

    usage: Final = derivations.usage(args)
    cost: Final = derivations.cost(args)
    timing: Final = derivations.attempt_timing(args)
    state: Final = derivations.rate_limit_state(args)

    assert usage is not None
    assert usage.logged == UsageRecord(prompt_tokens="ten", completion_tokens=float("inf"), rejected=("total_tokens",))
    assert cost == Cost(response_cost=0.5, zero_cost=ZeroCost(reason_present=False))
    assert timing is not None
    assert (timing.queue_seconds, timing.overhead_ms) == ("slow", "slow")
    assert state == RateLimitState(
        key_model_group="g",
        key_remaining_requests="250",
        limiter_headers={"x-ratelimit-key-remaining": True},
        provider_remaining_requests=" 77 ",
        hidden_params_present=True,
        additional_headers_present=True,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ((), None),
        (
            {"guardrail_mode": "pre_call", "duration": "0.05"},
            (GuardrailEntry(mapping=True, modes=("pre_call",), duration="0.05"),),
        ),
        (
            [{"guardrail_mode": ["pre_call", 7, {"x": 1}]}, "text"],
            (GuardrailEntry(mapping=True, modes=("pre_call", 7, None)), GuardrailEntry(mapping=False)),
        ),
        ("abc", (GuardrailEntry(mapping=False),)),
    ],
    ids=("none", "empty", "single-mapping", "list", "truthy-not-a-list"),
)
def test_guardrail_info_keeps_what_prometheus_iterates(
    raw: object, expected: tuple[GuardrailEntry, ...] | None
) -> None:
    payload: Final = {"guardrail_information": raw}

    assert derivations.guardrail_info(HookArgs(request={"standard_logging_object": payload})) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({}, Deployment(params_present=True)),
        (
            {"id": "m1", "tpm": "100", "rpm": 5.5},
            Deployment(params_present=True, model_info=ModelInfo(id="m1", tpm="100", rpm=5.5)),
        ),
        ("info", Deployment(params_present=True, rejected=("model_info",))),
    ],
    ids=("empty-is-falsy", "mapping", "truthy-not-a-mapping"),
)
def test_deployment_model_info_keeps_python_truthiness(raw: object, expected: Deployment) -> None:
    request: Final = {"litellm_params": {"metadata": {"model_info": raw}}}

    assert derivations.deployment(HookArgs(request=request)) == expected


def test_request_tags_keep_scalar_tags_and_mark_others_as_never_matching() -> None:
    payload: Final = {"request_tags": ["prod", 7, {"a": 1}, None]}

    assert derivations.request_tags(HookArgs(request={"standard_logging_object": payload})) == ("prod", 7, None, None)


def test_a_completion_start_that_is_present_but_none_differs_from_a_missing_one() -> None:
    present: Final = derivations.attempt_timing(HookArgs(request={"completion_start_time": None}))
    missing: Final = derivations.attempt_timing(HookArgs(request={}))

    assert present is not None
    assert missing is not None
    assert (present.completion_start_present, missing.completion_start_present) == (True, False)
