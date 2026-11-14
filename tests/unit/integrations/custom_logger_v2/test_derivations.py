import json
import math
import re
from collections.abc import Callable, Generator, Mapping
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace, UnionType
from typing import Annotated, Final, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, TypeAdapter

import litellm
from litellm.integrations.custom_logger_v2 import derivations
from litellm.integrations.custom_logger_v2.derivations import RECORD_TYPES
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    get_model_group_from_litellm_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the helper takes an untyped dict
)
from litellm.proxy.hooks import dynamic_rate_limiter_v3
from litellm.types.integrations.custom_logger_v2 import (
    AgentPolicy,
    AttemptTiming,
    BudgetPlan,
    BudgetPlanCounter,
    BudgetReservation,
    Cost,
    Deployment,
    EstimatedOutputTokens,
    GuardrailEntry,
    InputSequenceLength,
    KeyIdentity,
    KeyLimits,
    LoggedKeyIdentity,
    LoggedModelGroup,
    MetadataKind,
    MetadataLabel,
    ModelInfo,
    Moment,
    Outcome,
    RateLimitState,
    Record,
    ReservationEntry,
    Scalar,
    Stringified,
    Usage,
    UsageRecord,
    ZeroCost,
    coerce_record,
    reset_coercion_reports_for_tests,
)
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues, get_input_sequence_length_bucket

RECORDS_JSON: Final = Path(__file__).parents[4] / "litellm/rust/crates/callbacks/records.json"


class _Unprintable:
    def __str__(self) -> str:
        raise RuntimeError("no text")


class _Undecided:
    def __bool__(self) -> bool:
        raise RuntimeError("no truth")


class _HashRaises:
    def __hash__(self) -> int:
        raise RuntimeError("no hash")


class _Falsy:
    def __str__(self) -> str:
        return "falsy object"

    def __bool__(self) -> bool:
        return False


_SCALARS: Final[Mapping[object, object]] = {
    Scalar: 42,
    str: "text",
    bool: True,
    int: 7,
    float: 1.5,
    tuple[str, ...]: ["member"],
}
_WRONG: Final[Mapping[object, object]] = {
    Scalar: _Unprintable(),
    str: 7,
    bool: "maybe",
    int: "seven",
    float: "one and a half",
}
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


_CARRIED: Final = (
    pytest.param(
        ["a1"],
        Stringified(
            text="['a1']",
            truthy=True,
            hashable=False,
            integer=False,
            type_name="list",
            hash_error="unhashable type: 'list'",
        ),
        id="list",
    ),
    pytest.param(
        [],
        Stringified(
            text="[]",
            truthy=False,
            hashable=False,
            integer=False,
            type_name="list",
            hash_error="unhashable type: 'list'",
        ),
        id="empty_list",
    ),
    pytest.param(
        {"a": 1},
        Stringified(
            text="{'a': 1}",
            truthy=True,
            hashable=False,
            integer=False,
            type_name="dict",
            hash_error="unhashable type: 'dict'",
        ),
        id="dict",
    ),
    pytest.param(
        {},
        Stringified(
            text="{}",
            truthy=False,
            hashable=False,
            integer=False,
            type_name="dict",
            hash_error="unhashable type: 'dict'",
        ),
        id="empty_dict",
    ),
    pytest.param(
        {"a"},
        Stringified(
            text="{'a'}",
            truthy=True,
            hashable=False,
            integer=False,
            type_name="set",
            hash_error="unhashable type: 'set'",
        ),
        id="set",
    ),
    pytest.param(
        set(),
        Stringified(
            text="set()",
            truthy=False,
            hashable=False,
            integer=False,
            type_name="set",
            hash_error="unhashable type: 'set'",
        ),
        id="empty_set",
    ),
    pytest.param(
        ("a",),
        Stringified(text="('a',)", truthy=True, hashable=True, integer=False, type_name="tuple", hash_error=None),
        id="tuple",
    ),
    pytest.param(
        ("a", ["b"]),
        Stringified(
            text="('a', ['b'])",
            truthy=True,
            hashable=False,
            integer=False,
            type_name="tuple",
            hash_error="unhashable type: 'list'",
        ),
        id="tuple_of_list",
    ),
    pytest.param(
        frozenset(),
        Stringified(
            text="frozenset()", truthy=False, hashable=True, integer=False, type_name="frozenset", hash_error=None
        ),
        id="frozenset",
    ),
    pytest.param(
        b"7",
        Stringified(text="b'7'", truthy=True, hashable=True, integer=False, type_name="bytes", hash_error=None),
        id="bytes",
    ),
    pytest.param(
        b"",
        Stringified(text="b''", truthy=False, hashable=True, integer=False, type_name="bytes", hash_error=None),
        id="empty_bytes",
    ),
    pytest.param(
        _Falsy(),
        Stringified(
            text="falsy object", truthy=False, hashable=True, integer=False, type_name="_Falsy", hash_error=None
        ),
        id="object",
    ),
    pytest.param(
        2**127,
        Stringified(
            text="170141183460469231731687303715884105728",
            truthy=True,
            hashable=True,
            integer=True,
            type_name="int",
            hash_error=None,
        ),
        id="past_i128",
    ),
    pytest.param(
        -(2**127) - 1,
        Stringified(
            text="-170141183460469231731687303715884105729",
            truthy=True,
            hashable=True,
            integer=True,
            type_name="int",
            hash_error=None,
        ),
        id="below_i128",
    ),
)


@pytest.mark.parametrize(("raw", "carried"), _CARRIED)
def test_a_scalar_member_carries_any_other_value_as_the_str_text_and_truthiness_python_reads(
    raw: object, carried: Stringified, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="LiteLLM"):
        record: Final = coerce_record(UsageRecord, {"prompt_tokens": raw})

    assert record == UsageRecord(prompt_tokens=carried)
    text: Final = carried.text
    assert record.prompt_tokens != text
    assert not isinstance(record.prompt_tokens, str)
    assert caplog.records == []


def _priority_lookup_raises(priority: object) -> bool:
    try:
        _ = priority in {"gold": 0.5}
    except TypeError:
        return True
    return False


def _descriptor_identities_raise(value: object) -> bool:
    try:
        _ = dict.fromkeys((("team", value),))
    except TypeError:
        return True
    return False


@pytest.mark.parametrize(("raw", "carried"), _CARRIED)
def test_hashable_says_whether_capacity_and_the_limiter_raise_when_they_hash_the_value(
    raw: object, carried: Stringified
) -> None:
    identity: Final = coerce_record(KeyIdentity, {"team_priority": raw, "org_id": raw})

    assert identity.team_priority == identity.org_id == carried
    assert carried.hashable is not _priority_lookup_raises(raw)
    assert carried.hashable is not _descriptor_identities_raise(raw)
    assert carried.integer is isinstance(raw, int)


def _raised(call: Callable[[], object]) -> str | None:
    try:
        _ = call()
    except TypeError as error:
        return str(error)
    return None


def _match_error_from_type_name(type_name: str) -> str:
    if type_name in ("bytes", "bytearray", "memoryview"):
        return "cannot use a string pattern on a bytes-like object"
    return f"expected string or bytes-like object, got '{type_name}'"


@pytest.mark.parametrize(("raw", "carried"), _CARRIED)
def test_a_stringified_value_carries_the_error_text_python_raises_when_it_hashes_or_matches_it(
    raw: object, carried: Stringified
) -> None:
    identity: Final = coerce_record(KeyIdentity, {"team_priority": raw})

    assert identity.team_priority == carried
    assert carried.hash_error == _raised(lambda: raw in {"gold": 0.5})
    assert carried.hash_error == _raised(lambda: dict.fromkeys((("team", raw),)))
    assert _raised(lambda: re.match("prod.*", raw)) == _match_error_from_type_name(carried.type_name)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownLambdaType]  # what prometheus does with a non-str tag


@pytest.mark.parametrize(
    ("raw", "type_name"),
    [
        (list[object](), "list"),
        (dict[str, object](), "dict"),
        (set[object](), "set"),
        (tuple[object, ...](), "tuple"),
        (frozenset[object](), "frozenset"),
        (_Falsy(), "_Falsy"),
    ],
    ids=("list", "dict", "set", "tuple", "frozenset", "object"),
)
def test_a_stringified_value_carries_the_type_name_float_names_when_it_refuses_it(raw: object, type_name: str) -> None:
    carried: Final = coerce_record(UsageRecord, {"prompt_tokens": raw}).prompt_tokens

    assert isinstance(carried, Stringified)
    assert carried.type_name == type_name
    assert _raised(lambda: float(raw)) == f"float() argument must be a string or a real number, not '{type_name}'"  # pyright: ignore[reportArgumentType]  # what prometheus and capacity do with a non-number


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(_Unprintable(), id="str_raises"),
        pytest.param(_Undecided(), id="bool_raises"),
        pytest.param(10**5000, id="int_too_long_for_str"),
        pytest.param(_HashRaises(), id="hash_raises_other_than_type_error"),
    ],
)
def test_a_value_whose_str_or_bool_raises_is_named_in_rejected(raw: object) -> None:
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


def _built_reservation() -> dict[str, object]:  # mutable-ok: budget_reservation.py mutates the dict it builds
    return {
        "reserved_cost": 0.75,
        "entries": [
            {
                "counter_key": "spend:key:hash",
                "entity_type": "Key",
                "entity_id": "hash",
                "reserved_cost": 0.75,
                "applied_adjustment": 0.0,
            },
            {
                "counter_key": "spend:model_access_group:g1",
                "entity_type": "Model access group",
                "entity_id": "g1",
                "reserved_cost": 0.75,
                "applied_adjustment": 0.0,
            },
        ],
        "finalized": False,
        "callback_bound": False,
        "input_cost": 0.25,
        "input_tokens": 40,
        "rust_call_id": "call-1",
    }


def test_the_reservation_dict_budget_reservation_py_builds_reads_with_its_extra_keys_and_every_entry() -> None:
    auth: Final = _auth()
    auth.budget_reservation = _built_reservation()

    read: Final = derivations.budget_reservation(HookArgs(auth=auth))

    assert read is not None
    assert read.rejected is None
    assert (read.reserved_cost, read.finalized, read.callback_bound, read.input_cost, read.input_tokens) == (
        0.75,
        False,
        False,
        0.25,
        40,
    )
    assert read.entries is not None
    reserved_groups: Final = frozenset(
        entry.entity_id for entry in read.entries if entry is not None and entry.entity_type == "Model access group"
    )
    assert reserved_groups == frozenset({"g1"})
    assert read.entries[0] == ReservationEntry(
        counter_key="spend:key:hash", entity_type="Key", entity_id="hash", reserved_cost=0.75, applied_adjustment=0.0
    )


@pytest.mark.parametrize(
    ("entries", "expected", "is_list", "rejected"),
    [
        pytest.param(
            ({"counter_key": "k"}, "a list"),
            (ReservationEntry(counter_key="k"), None),
            False,
            None,
            id="tuple_budgets_iterates_and_capacity_reads_as_covered",
        ),
        pytest.param("ab", (None, None), False, None, id="text_iterates_as_characters"),
        pytest.param({"counter_key": "k"}, (None,), False, None, id="dict_iterates_as_keys"),
        pytest.param(0, (), False, None, id="falsy_is_the_empty_list_budgets_uses"),
        pytest.param(5, None, None, ("entries",), id="truthy_that_does_not_iterate"),
        pytest.param([], (), True, None, id="empty_list"),
        pytest.param(
            ["k", {"counter_key": "k"}], (None, ReservationEntry(counter_key="k")), True, None, id="text_entry"
        ),
        pytest.param(
            [{"counter_key": 7, "extra": 1}],
            (ReservationEntry(counter_key="7"),),
            True,
            None,
            id="key_formatted_by_str",
        ),
        pytest.param(
            [{"entity_type": ["x"], "entity_id": "g"}],
            (ReservationEntry(entity_id="g", rejected=("entity_type",)),),
            True,
            None,
            id="non_text_entity_type",
        ),
    ],
)
def test_reservation_entries_keep_what_capacity_and_budgets_each_read(
    entries: object,
    expected: tuple[ReservationEntry | None, ...] | None,
    is_list: bool | None,
    rejected: tuple[str, ...] | None,
) -> None:
    auth: Final = _auth()
    auth.budget_reservation = {"reserved_cost": 1.0, "entries": entries}

    read: Final = derivations.budget_reservation(HookArgs(auth=auth))

    assert read is not None
    assert (read.entries, read.entries_is_list, read.rejected) == (expected, is_list, rejected)


def test_a_reservation_write_never_stores_the_entries_shape_it_read() -> None:
    reservation: Final[dict[str, object]] = {"entries": ({"counter_key": "k"},)}  # mutable-ok: the write updates it
    auth: Final = _auth()
    auth.budget_reservation = reservation
    args: Final = HookArgs(auth=auth)
    read: Final = derivations.budget_reservation(args)
    assert read is not None
    assert read.entries_is_list is False

    derivations.write_budget_reservation(args, read.model_copy(update={"finalized": True}))
    derivations.write_budget_reservation(args, {"entries_is_list": True, "input_tokens": 3})

    assert reservation == {"entries": [{"counter_key": "k"}], "finalized": True, "input_tokens": 3}


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        pytest.param("0.5", 0.5, id="numeric_text"),
        pytest.param(0, 0.0, id="zero"),
        pytest.param("", 0.0, id="falsy_text_is_zero"),
        pytest.param(True, 1.0, id="flag"),
        pytest.param(float("inf"), float("inf"), id="infinity"),
        pytest.param("-inf", float("-inf"), id="negative_infinity_text"),
    ],
)
def test_a_reservation_cost_reads_as_the_float_budgets_computes(raw: object, value: float) -> None:
    auth: Final = _auth()
    auth.budget_reservation = {"reserved_cost": raw, "input_cost": raw, "entries": [{"applied_adjustment": raw}]}

    read: Final = derivations.budget_reservation(HookArgs(auth=auth))

    assert read is not None
    assert read.entries is not None
    entry: Final = read.entries[0]
    assert entry is not None
    assert (read.reserved_cost, read.input_cost, entry.applied_adjustment) == (value, value, value)


def test_a_nan_reservation_cost_reads_as_nan() -> None:
    auth: Final = _auth()
    auth.budget_reservation = {"reserved_cost": float("nan")}

    read: Final = derivations.budget_reservation(HookArgs(auth=auth))

    assert read is not None
    assert read.reserved_cost is not None
    assert math.isnan(read.reserved_cost)


@pytest.mark.parametrize("flag", [1, "true", "yes"])
def test_a_finalized_or_bound_flag_that_is_not_a_bool_is_named_in_rejected_because_python_tests_is_true(
    flag: object,
) -> None:
    auth: Final = _auth()
    auth.budget_reservation = {"finalized": flag, "callback_bound": flag}

    read: Final = derivations.budget_reservation(HookArgs(auth=auth))

    assert read == BudgetReservation(rejected=("finalized", "callback_bound"))


def test_an_infinite_reservation_write_lands_in_the_shared_dict_as_floats_with_entries_as_a_list() -> None:
    reservation: Final = _built_reservation()
    auth: Final = _auth()
    auth.budget_reservation = reservation
    args: Final = HookArgs(auth=auth)
    read: Final = derivations.budget_reservation(args)
    assert read is not None
    assert read.entries is not None

    derivations.write_budget_reservation(
        args,
        read.model_copy(
            update={
                "reserved_cost": float("inf"),
                "input_cost": float("nan"),
                "entries": (read.entries[0], None),
                "rejected": ("finalized",),
            }
        ),
    )

    assert reservation["reserved_cost"] == float("inf")
    input_cost: Final = reservation["input_cost"]
    assert isinstance(input_cost, float)
    assert math.isnan(input_cost)
    assert reservation["entries"] == [
        {
            "counter_key": "spend:key:hash",
            "entity_type": "Key",
            "entity_id": "hash",
            "reserved_cost": 0.75,
            "applied_adjustment": 0.0,
        },
        None,
    ]
    assert "rejected" not in reservation
    assert reservation["rust_call_id"] == "call-1"


def test_a_wire_reservation_write_drops_nulls_inside_entries_and_the_rejected_names() -> None:
    reservation: Final[dict[str, object]] = {"reserved_cost": 0.5}  # mutable-ok: the write updates this dict in place
    auth: Final = _auth()
    auth.budget_reservation = reservation

    derivations.write_budget_reservation(
        HookArgs(auth=auth),
        {
            "rejected": None,
            "reserved_cost": float("inf"),
            "entries": [{"rejected": ["entity_type"], "counter_key": "k", "reserved_cost": None}],
        },
    )

    assert reservation == {"reserved_cost": float("inf"), "entries": [{"counter_key": "k"}]}
    entries: Final = TypeAdapter(list[object]).validate_python(reservation["entries"])
    assert type(entries[0]) is dict


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


def test_key_identity_carries_a_list_priority_and_agent_id_as_the_text_python_formats() -> None:
    auth: Final = _auth(team_metadata={"priority": ["gold"]})
    auth.agent_id = ["a1"]  # pyright: ignore[reportAttributeAccessIssue]  # the shape a hand-built auth object can hold

    identity: Final = derivations.key_identity(HookArgs(auth=auth))

    assert identity is not None
    assert identity.rejected is None
    assert (identity.agent_id, identity.team_priority) == (
        Stringified(
            text="['a1']",
            truthy=True,
            hashable=False,
            integer=False,
            type_name="list",
            hash_error="unhashable type: 'list'",
        ),
        Stringified(
            text="['gold']",
            truthy=True,
            hashable=False,
            integer=False,
            type_name="list",
            hash_error="unhashable type: 'list'",
        ),
    )


@pytest.mark.parametrize(
    "raw",
    [[{"model": "gpt", "alternates": ["mini"]}], [{"model": "gpt", "alternates": [7]}], {"model": "gpt"}, "gpt"],
    ids=("well-formed", "alternate-not-a-string", "not-a-list", "text"),
)
def test_a_team_model_preferences_entry_is_ignored_like_the_capacity_callback_ignores_it(
    raw: object, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("DEBUG"):
        identity: Final = derivations.key_identity(HookArgs(auth=_auth(team_metadata={"model_preferences": raw})))

    assert identity == derivations.key_identity(HookArgs(auth=_auth(team_metadata={})))
    assert identity is not None
    assert identity.rejected is None
    assert "model_preferences" not in KeyIdentity.model_fields
    assert not hasattr(dynamic_rate_limiter_v3, "alternates_for")
    assert not [entry for entry in caplog.records if "model_preferences" in entry.getMessage()]


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
    assert usage.logged == UsageRecord(
        prompt_tokens="ten",
        completion_tokens=float("inf"),
        total_tokens=Stringified(
            text="[10]",
            truthy=True,
            hashable=False,
            integer=False,
            type_name="list",
            hash_error="unhashable type: 'list'",
        ),
    )
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
            (
                GuardrailEntry(
                    mapping=True,
                    modes=(
                        "pre_call",
                        7,
                        Stringified(
                            text="{'x': 1}",
                            truthy=True,
                            hashable=False,
                            integer=False,
                            type_name="dict",
                            hash_error="unhashable type: 'dict'",
                        ),
                    ),
                ),
                GuardrailEntry(mapping=False),
            ),
        ),
        ("abc", (GuardrailEntry(mapping=False),)),
        (
            {"guardrail_mode": ("pre_call",)},
            (
                GuardrailEntry(
                    mapping=True,
                    modes=(
                        Stringified(
                            text="('pre_call',)",
                            truthy=True,
                            hashable=True,
                            integer=False,
                            type_name="tuple",
                            hash_error=None,
                        ),
                    ),
                ),
            ),
        ),
        ({"guardrail_mode": 7}, (GuardrailEntry(mapping=True, modes=(7,)),)),
        ({"duration": 0.5}, (GuardrailEntry(mapping=True, modes=(None,), duration=0.5),)),
    ],
    ids=(
        "none",
        "empty",
        "single-mapping",
        "list",
        "truthy-not-a-list",
        "tuple-mode-is-one-mode",
        "int-mode-is-one-mode",
        "missing-mode-is-one-none-mode",
    ),
)
def test_guardrail_info_keeps_what_prometheus_iterates(
    raw: object, expected: tuple[GuardrailEntry, ...] | None
) -> None:
    payload: Final = {"guardrail_information": raw}

    assert derivations.guardrail_info(HookArgs(request={"standard_logging_object": payload})) == expected


@pytest.mark.parametrize(
    "mode",
    [("pre_call",), ["pre_call"], ["pre_call", "during_call"], "post_call", 7, None, ["pre_call", 7]],
    ids=repr,
)
def test_guardrail_modes_select_the_str_modes_prometheus_counts_as_additive(mode: object) -> None:
    from litellm.integrations.prometheus import PrometheusLogger

    payload: Final = {"guardrail_information": {"guardrail_mode": mode}}
    entries: Final = derivations.guardrail_info(HookArgs(request={"standard_logging_object": payload}))
    assert entries is not None
    modes: Final = frozenset(item for item in entries[0].modes or () if isinstance(item, str))

    assert (bool(modes) and modes <= PrometheusLogger._ADDITIVE_GUARDRAIL_MODES) is (
        PrometheusLogger._guardrail_is_additive({"guardrail_mode": mode})  # pyright: ignore[reportArgumentType]  # the shapes guardrails write
    )


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


def test_request_tags_keep_scalar_tags_and_carry_others_as_their_text() -> None:
    payload: Final = {"request_tags": ["prod", 7, {"a": 1}, [], None]}

    assert derivations.request_tags(HookArgs(request={"standard_logging_object": payload})) == (
        "prod",
        7,
        Stringified(
            text="{'a': 1}",
            truthy=True,
            hashable=False,
            integer=False,
            type_name="dict",
            hash_error="unhashable type: 'dict'",
        ),
        Stringified(
            text="[]",
            truthy=False,
            hashable=False,
            integer=False,
            type_name="list",
            hash_error="unhashable type: 'list'",
        ),
        None,
    )


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        pytest.param(lambda: "ab", ("a", "b"), id="text_iterates_as_characters"),
        pytest.param(lambda: {"prod": 1, "dev": 2}, ("prod", "dev"), id="dict_iterates_as_keys"),
        pytest.param(lambda: {"prod"}, ("prod",), id="set"),
        pytest.param(lambda: (tag for tag in ("prod", "dev")), ("prod", "dev"), id="generator"),
        pytest.param(lambda: ("prod", 7), ("prod", 7), id="tuple"),
        pytest.param(lambda: b"a", (97,), id="bytes_iterate_as_ints"),
    ],
)
def test_request_tags_carry_any_iterable_as_the_items_prometheus_iterates(
    make: Callable[[], object], expected: tuple[object, ...]
) -> None:
    payload: Final = {"request_tags": make()}

    assert derivations.request_tags(HookArgs(request={"standard_logging_object": payload})) == expected
    assert UserAPIKeyLabelValues(tags=make()).tags == expected


def test_request_tags_that_do_not_iterate_are_absent_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="LiteLLM"):
        tags: Final = derivations.request_tags(HookArgs(request={"standard_logging_object": {"request_tags": 7}}))

    assert tags is None
    assert [entry.getMessage().split(":")[1].strip() for entry in caplog.records] == [
        "request_tags.value holds a value the contract cannot carry, so it is named in rejected"
    ]


def test_a_completion_start_that_is_present_but_none_differs_from_a_missing_one() -> None:
    present: Final = derivations.attempt_timing(HookArgs(request={"completion_start_time": None}))
    missing: Final = derivations.attempt_timing(HookArgs(request={}))

    assert present is not None
    assert missing is not None
    assert (present.completion_start_present, missing.completion_start_present) == (True, False)


def _helper_outcome(request: dict[str, object]) -> LoggedModelGroup:
    try:
        group: Final = get_model_group_from_litellm_kwargs(request)
    except Exception as error:  # noqa: BLE001  # the outcome the logging handlers see
        return LoggedModelGroup(error_type=type(error).__name__, error_text=str(error))
    return coerce_record(LoggedModelGroup, {"value": group})


_MODEL_GROUP_REQUESTS: Final = (
    pytest.param({"litellm_params": {"metadata": {"model_group": "g"}}}, id="metadata"),
    pytest.param(
        {"litellm_params": {"metadata": {"model_group": "g"}, "litellm_metadata": {"model_group": "lg"}}},
        id="litellm_metadata_only_inside_params",
    ),
    pytest.param(
        {
            "litellm_metadata": None,
            "litellm_params": {"metadata": {"model_group": "g"}, "litellm_metadata": {"model_group": "lg"}},
        },
        id="litellm_metadata_top_level",
    ),
    pytest.param({"litellm_metadata": {}, "litellm_params": {"metadata": {"model_group": "g"}}}, id="empty_bucket"),
    pytest.param({"litellm_params": {"metadata": {"model_group": 7}}}, id="int_group"),
    pytest.param({"litellm_params": {"metadata": {"model_group": ["g"]}}}, id="list_group"),
    pytest.param({"litellm_params": ["g"]}, id="params_not_a_mapping"),
    pytest.param({"litellm_params": {"metadata": "g"}}, id="metadata_not_a_mapping"),
    pytest.param({"litellm_params": {"metadata": []}}, id="falsy_metadata"),
    pytest.param({}, id="no_params"),
)


@pytest.mark.parametrize("request_data", _MODEL_GROUP_REQUESTS)
def test_logged_model_group_is_what_the_logging_handlers_helper_returns_or_raises(
    request_data: dict[str, object],
) -> None:
    assert derivations.logged_model_group(HookArgs(request=request_data)) == _helper_outcome(request_data)


def test_logged_model_group_names_the_error_the_helper_raises() -> None:
    assert derivations.logged_model_group(HookArgs(request={"litellm_params": {"metadata": "g"}})) == LoggedModelGroup(
        error_type="AttributeError", error_text="'str' object has no attribute 'get'"
    )


def test_rate_limit_state_reads_remaining_limits_under_the_model_group_the_helper_picks() -> None:
    request: Final = {
        "litellm_params": {
            "litellm_metadata": {"model_group": "lg"},
            "metadata": {
                "model_group": "g",
                "litellm-key-remaining-requests-g": 5,
                "litellm-key-remaining-tokens-g": 6,
                "litellm-key-remaining-requests-lg": 50,
            },
        }
    }

    state: Final = derivations.rate_limit_state(HookArgs(request=request))

    assert state is not None
    assert (state.key_model_group, state.key_remaining_requests, state.key_remaining_tokens) == ("g", 5, 6)


def test_rate_limit_state_formats_a_missing_or_non_str_model_group_into_the_key_like_prometheus() -> None:
    request: Final = {
        "litellm_params": {
            "metadata": {
                "model_group": ["g"],
                "litellm-key-remaining-requests-['g']": 5,
                "litellm-key-remaining-tokens-None": 6,
            }
        }
    }
    missing: Final = {"litellm_params": {"metadata": {"litellm-key-remaining-tokens-None": 6}}}

    state: Final = derivations.rate_limit_state(HookArgs(request=request))
    without_group: Final = derivations.rate_limit_state(HookArgs(request=missing))

    assert state is not None
    assert without_group is not None
    assert (state.key_remaining_requests, state.key_remaining_tokens) == (5, None)
    assert (without_group.key_model_group, without_group.key_remaining_tokens) == (None, 6)


def test_rate_limit_state_leaves_the_key_limits_absent_when_the_helper_raises() -> None:
    state: Final = derivations.rate_limit_state(HookArgs(request={"litellm_params": {"metadata": "g"}}))

    assert state == RateLimitState(hidden_params_present=False, additional_headers_present=False)


def _capacity_reservation_check_raises(reservation: object) -> str | None:
    try:
        _ = reservation.get("entries")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # what _reservation_covers does
    except AttributeError as error:
        return str(error)
    return None


def test_metadata_labels_are_the_str_values_prometheus_labels_in_config_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "custom_prometheus_metadata_labels",
        ["region", "metadata.team_tier", "nested.a", "user_api_key_team_alias", "absent", "is_silent_experiment"],
    )
    payload: Final = {
        "metadata": {
            "team_tier": "gold",
            "user_api_key_team_alias": 7,
            "is_silent_experiment": True,
            "requester_metadata": {"region": "eu", "nested": {"a": "b"}},
        }
    }

    labels: Final = derivations.metadata_labels(HookArgs(request={"standard_logging_object": payload}))

    assert labels == (
        MetadataLabel(name="region", value="eu"),
        MetadataLabel(name="metadata_team_tier", value="gold"),
        MetadataLabel(name="nested_a", value="b"),
        MetadataLabel(name="is_silent_experiment", value="true"),
    )


@pytest.mark.parametrize("payload", [None, "payload", {"metadata": ["x"]}], ids=repr)
def test_metadata_labels_are_empty_where_prometheus_finds_no_metadata(
    payload: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["region"])

    assert derivations.metadata_labels(HookArgs(request={"standard_logging_object": payload})) == ()


class _Upstream:
    total_tokens: int = 12


@pytest.mark.parametrize(
    ("payload", "extra", "result", "prompt_tokens"),
    [
        pytest.param({"prompt_tokens": 4000, "metadata": {}}, {}, None, 4000, id="payload_tokens"),
        pytest.param(
            {"prompt_tokens": 0, "metadata": {}},
            {"combined_usage_object": {"prompt_tokens": 3}, "_litellm_upstream_reported_usage": _Upstream()},
            None,
            None,
            id="upstream_reported",
        ),
        pytest.param({"prompt_tokens": 0, "metadata": {}}, {}, None, None, id="no_usage"),
        pytest.param(
            {"prompt_tokens": 0, "metadata": {"usage_object": {"prompt_tokens": 1500}}},
            {},
            {"usage": {"prompt_tokens": 7}},
            1500,
            id="usage_object",
        ),
        pytest.param(
            {"prompt_tokens": 0, "metadata": {}}, {}, {"usage": {"prompt_tokens": 900}}, 900, id="normalized_response"
        ),
    ],
)
def test_input_sequence_length_is_the_bucket_prometheus_picks_from_payload_kwargs_and_response(
    payload: dict[str, object], extra: dict[str, object], result: object, prompt_tokens: int | None
) -> None:
    request: Final = {"standard_logging_object": payload, **extra}

    served: Final = derivations.input_sequence_length(HookArgs(request=request, result=result))

    assert served == InputSequenceLength(bucket=get_input_sequence_length_bucket(prompt_tokens))
    assert served is not None
    assert served.bucket == PrometheusLogger._get_input_sequence_length(payload, request, result)  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # prometheus's own rule


def test_input_sequence_length_names_the_error_prometheus_raises() -> None:
    request: Final = {"standard_logging_object": {"prompt_tokens": 0, "metadata": None}}

    assert derivations.input_sequence_length(
        HookArgs(request=request, result={"usage": {"prompt_tokens": 9}})
    ) == InputSequenceLength(error_type="AttributeError", error_text="'NoneType' object has no attribute 'get'")


def _moment_at(second: int) -> Moment:
    return Moment(
        year=2026, month=11, day=13, hour=9, minute=30, second=second, microsecond=250, utc_offset_seconds=-18000
    )


def test_attempt_timing_carries_the_times_prometheus_reads_apart_from_the_handler_arguments() -> None:
    request: Final = {"start_time": _MOMENT, "end_time": _MOMENT + timedelta(seconds=9)}

    timing: Final = derivations.attempt_timing(
        HookArgs(request=request, start_time=_MOMENT + timedelta(seconds=5), end_time=_MOMENT + timedelta(seconds=7))
    )

    assert timing is not None
    assert (timing.start, timing.end, timing.logged_start, timing.logged_end) == (
        _moment_at(20),
        _moment_at(22),
        _moment_at(15),
        _moment_at(24),
    )


@pytest.mark.parametrize("end_time", [None, 0, ""], ids=repr)
def test_a_falsy_logged_end_is_absent_so_the_port_falls_back_to_now_like_prometheus(end_time: object) -> None:
    timing: Final = derivations.attempt_timing(HookArgs(request={"end_time": end_time}))

    assert timing is not None
    assert (timing.logged_end, timing.rejected) == (None, None)


def test_key_limits_carry_the_agent_policy_limits_the_limiter_reads_raw() -> None:
    auth: Final = _auth()
    auth.managed_agent_policy = SimpleNamespace(rpm_limit=5, tpm_limit="9", session_rpm_limit=None)  # pyright: ignore[reportAttributeAccessIssue]  # the limiter reads the policy with getattr
    auth.invoked_agent_policy = SimpleNamespace(session_tpm_limit=[1])  # pyright: ignore[reportAttributeAccessIssue]  # the limiter reads the policy with getattr

    limits: Final = derivations.key_limits(HookArgs(auth=auth))

    assert limits is not None
    assert (limits.managed_agent_policy, limits.invoked_agent_policy) == (
        AgentPolicy(rpm_limit=5, tpm_limit="9"),
        AgentPolicy(
            session_tpm_limit=Stringified(
                text="[1]",
                truthy=True,
                hashable=False,
                integer=False,
                type_name="list",
                hash_error="unhashable type: 'list'",
            )
        ),
    )
    assert derivations.key_limits(HookArgs(auth=_auth())) == KeyLimits(estimated_output_tokens=EstimatedOutputTokens())
