import contextvars
import copy
import json
import math
from collections.abc import Awaitable, Callable, Generator, Mapping
from datetime import datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter

import litellm
from litellm.exceptions import BudgetExceededError
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import (
    Abort,
    Activation,
    CallFacts,
    CallIds,
    CallKind,
    CallTarget,
    CallTiming,
    Field,
    Hook,
    Patch,
    build_manifest,
)
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.dispatcher import HookResult
from litellm.integrations.custom_logger_v2.host import (
    current_scope,
    get_dispatcher,
    request_scope,
    reset_host_for_tests,
)
from litellm.integrations.custom_logger_v2.legacy import (
    PROXY_REQUEST_KEYS,
    Admit,
    KwargsFields,
    clear_subscribers_cache_for_tests,
    hook_subscribers,
    kwargs_after,
    run_subscribers,
)
from litellm.integrations.custom_logger_v2.logger import CustomLoggerV2
from litellm.integrations.custom_logger_v2.rust import (
    ABORT_ERRORS,
    RustLogger,
    WireAbort,
    contract_table,
    rust_logger,
    register_abort_error,
    verdict,
)
from litellm.integrations.custom_logger_v2.offpath import InProcessSink
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS, Logging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import bind_v2_request_scope
from litellm.proxy.utils import raise_v2_abort
from litellm.types.integrations.custom_logger_v2 import (
    BudgetReservation,
    Cost,
    GuardrailEntry,
    KeyIdentity,
    Outcome,
    ReservationEntry,
    Stringified,
)

RUST: Final = Path(__file__).parents[4] / "litellm/rust/crates"
CONTRACT: Final = RUST / "callbacks/contract.json"
EXAMPLE_MANIFEST: Final = RUST / "callbacks-python/tests/round_trip/manifest.json"
_REQUEST: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
CALL: Final = CallFacts(
    ids=CallIds(call_id="call-1"),
    kind=CallKind(call_type="acompletion"),
    target=CallTarget(model="gpt"),
    timing=CallTiming(start=1.0),
)


@pytest.fixture(autouse=True)
def fresh_host() -> Generator[None]:
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()
    yield
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()


class Handle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, object], Mapping[str, object] | None, object]] = []
        self.states: list[dict[str, object]] = []

    def manifest(self) -> str:
        return EXAMPLE_MANIFEST.read_text()

    def request_state(self) -> object:
        state: Final[dict[str, object]] = {}
        self.states.append(state)
        return state

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, /
    ) -> Awaitable[object]:
        self.calls.append((hook, fields, call, state))
        return self._answer(hook, fields, state)

    async def _answer(self, hook: str, fields: Mapping[str, object], state: object) -> object:
        model: Final = fields.get("model")
        if hook != "pre_call":
            return None
        if model == "blocked":
            return {"abort": {"status": 403, "message": "blocked", "code": "rust_block", "headers": {}, "error": None}}
        if isinstance(state, dict):
            state["model"] = model
        return {"patch": {"ops": [{"op": "set", "field": "model", "value": f"rust/{model}"}]}}


class Recorder(CustomLogger):
    def __init__(self, name: str, seen: list[str]) -> None:
        self.name = name
        self.seen = seen

    async def async_pre_call_hook(self, *args: object, **kwargs: object) -> None:
        return None

    async def async_log_success_event(self, *args: object, **kwargs: object) -> None:
        return None


def _binding(seen: list[str]) -> Callable[[object, object], Awaitable[None]]:
    def start(logger: object, data: object) -> Awaitable[None]:
        async def run() -> None:
            if isinstance(logger, Recorder):
                seen.append(f"{logger.name} {_REQUEST.validate_python(data).get('model')}")

        return run()

    return start


def test_the_python_enums_match_the_contract_rust_also_checks() -> None:
    assert contract_table() == json.loads(CONTRACT.read_text())


def test_a_rust_logger_takes_its_declarations_from_the_rust_manifest() -> None:
    logger: Final = rust_logger(Handle())
    manifest: Final = build_manifest(logger)

    assert isinstance(logger, RustLogger)
    assert isinstance(logger, CustomLoggerV2)
    assert manifest.selection.name == "rust_example"
    assert manifest.selection.activation is Activation.required
    assert set(manifest.subscriptions) == {Hook.pre_call, Hook.attempt_ended}
    pre_call: Final = manifest.subscriptions[Hook.pre_call]
    assert (pre_call.reads, pre_call.writes, pre_call.may_abort) == (
        frozenset({Field.model}),
        frozenset({Field.model}),
        True,
    )
    assert manifest.request_state is not None


def test_a_rust_built_twice_lands_in_an_event_list_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    first: Final = rust_logger(Handle())

    for logger in (first, rust_logger(Handle())):
        litellm.logging_callback_manager.add_litellm_async_success_callback(logger)

    assert litellm._async_success_callback == [first]


async def test_a_rust_logger_runs_at_its_position_between_two_legacy_loggers() -> None:
    seen: Final[list[str]] = []
    handle: Final = Handle()
    callbacks: Final[list[object]] = [Recorder("first", seen), rust_logger(handle), Recorder("last", seen)]
    data: Final[dict[str, object]] = {"model": "gpt"}

    result: Final = await run_subscribers(
        hook_subscribers(Hook.pre_call, callbacks), KwargsFields(data, PROXY_REQUEST_KEYS), _binding(seen), call=CALL
    )

    assert kwargs_after(result) is data
    assert seen == ["first gpt", "last rust/gpt"]
    assert handle.calls[0][1] == {"model": "gpt"}
    assert handle.calls[0][2] is not None
    assert handle.calls[0][2]["ids"] == {
        "call_id": "call-1",
        "trace_id": None,
        "session_id": None,
        "parent_call_id": None,
        "connection_id": None,
    }


async def test_a_rust_abort_stops_the_line_and_becomes_the_proxy_http_error() -> None:
    seen: Final[list[str]] = []
    callbacks: Final[list[object]] = [rust_logger(Handle()), Recorder("last", seen)]

    result: Final = await run_subscribers(
        hook_subscribers(Hook.pre_call, callbacks),
        KwargsFields({"model": "blocked"}, PROXY_REQUEST_KEYS),
        _binding(seen),
    )

    assert seen == []
    with pytest.raises(HTTPException) as raised:
        raise_v2_abort(result)
    assert raised.value.status_code == 403
    assert raised.value.detail == {"error": "blocked"}


async def test_a_projection_hands_rust_a_port_chosen_view_of_a_field() -> None:
    handle: Final = Handle()
    callbacks: Final[list[object]] = [rust_logger(handle, MappingProxyType({Field.model: lambda value: f"{value}!"}))]

    _ = await run_subscribers(
        hook_subscribers(Hook.pre_call, callbacks), KwargsFields({"model": "gpt"}, PROXY_REQUEST_KEYS), _binding([])
    )

    assert handle.calls[0][1] == {"model": "gpt!"}


def test_an_abort_carrying_a_registered_error_kind_raises_that_exception() -> None:
    def budget(wire: WireAbort) -> Exception:
        return BudgetExceededError(current_cost=2.0, max_budget=1.0, message=wire.message)

    register_abort_error("test_budget_exceeded", budget)
    try:
        abort: Final = verdict(
            {
                "abort": {
                    "status": 400,
                    "message": "over budget",
                    "code": None,
                    "headers": {"x-a": "1"},
                    "error": {"kind": "test_budget_exceeded", "fields": {}},
                }
            }
        )
    finally:
        _ = ABORT_ERRORS.pop("test_budget_exceeded")

    assert isinstance(abort, Abort)
    with pytest.raises(BudgetExceededError) as raised:
        raise_v2_abort(HookResult(fields={}, outcome=abort))
    assert raised.value.max_budget == 1.0


def test_an_abort_naming_an_unregistered_error_kind_logs_it_and_raises_the_generic_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wire: Final = {
        "status": 429,
        "message": "slow down",
        "code": None,
        "headers": {},
        "error": {"kind": "unknown", "fields": {}},
    }

    with caplog.at_level("ERROR", logger="LiteLLM"):
        abort: Final = verdict({"abort": wire})

    assert isinstance(abort, Abort)
    assert abort.exception is None
    assert [record.getMessage() for record in caplog.records] == [
        "no abort error is registered for kind 'unknown', raising the generic abort error"
    ]
    with pytest.raises(HTTPException) as raised:
        raise_v2_abort(HookResult(fields={}, outcome=abort))
    assert raised.value.status_code == 429


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        pytest.param(None, None, id="nothing"),
        pytest.param(
            {"patch": {"ops": [{"op": "merge", "field": "response_headers", "keys": {"x-a": "1"}}]}},
            Patch().merge(Field.response_headers, {"x-a": "1"}),
            id="merge",
        ),
    ],
)
def test_a_wire_verdict_becomes_the_step_0_outcome(wire: object, expected: Patch | None) -> None:
    assert verdict(wire) == expected


async def test_an_off_path_rust_hook_sees_the_state_its_pre_call_wrote_in_the_same_scope() -> None:
    handle: Final = Handle()
    callbacks: Final[list[object]] = [Recorder("first", []), rust_logger(handle)]
    sink: Final = get_dispatcher().sink

    with request_scope():
        _ = await run_subscribers(
            hook_subscribers(Hook.pre_call, callbacks), KwargsFields({"model": "gpt"}, PROXY_REQUEST_KEYS), _binding([])
        )
        _ = await run_subscribers(
            hook_subscribers(Hook.attempt_ended, callbacks, admit=Admit.every_entry),
            KwargsFields({"model": "rust/gpt"}, LOGGING_KEYS),
            _binding([]),
        )
    assert isinstance(sink, InProcessSink)
    await sink.drain()

    assert [hook for hook, *_ in handle.calls] == ["pre_call", "attempt_ended"]
    assert handle.calls[1][3] is handle.calls[0][3]
    assert handle.states == [{"model": "gpt"}]


def _bound(
    monkeypatch: pytest.MonkeyPatch, declared: frozenset[Hook], data: dict[str, object]
) -> tuple[object, object]:
    monkeypatch.setattr(CustomLoggerV2, "declared_hooks", declared)

    def bind() -> tuple[object, object]:
        scope = bind_v2_request_scope(data, UserAPIKeyAuth(), {})
        return scope, current_scope()

    return contextvars.copy_context().run(bind)


def test_no_scope_is_opened_while_no_v2_callback_class_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _bound(monkeypatch, frozenset(), {"model": "gpt"}) == (None, None)


def test_the_proxy_scope_is_bound_for_the_request_and_reaches_its_logging_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logging_obj: Final = Logging(
        model="gpt",
        messages=[],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="call-1",
        function_id="f",
    )
    scope, current = _bound(
        monkeypatch, frozenset({Hook.pre_call}), {"no-log": True, "litellm_logging_obj": logging_obj}
    )

    assert scope is not None
    assert current is scope
    assert logging_obj.v2_scope is scope
    assert isinstance(logging_obj.v2_auth, UserAPIKeyAuth)
    assert copy.deepcopy({"litellm_logging_obj": logging_obj})["litellm_logging_obj"].v2_scope is scope
    facts: Final = logging_obj.v2_call_facts()
    assert facts is not None
    assert (facts.ids.call_id, facts.kind.call_type, facts.target.model) == ("call-1", "acompletion", "gpt")


async def test_typed_records_reach_the_rust_with_the_python_values_json_mode_would_lose() -> None:
    handle: Final = Handle()
    logger: Final = rust_logger(handle)
    assert isinstance(logger, RustLogger)
    view: Final = SimpleNamespace(
        cost=Cost(response_cost=float("nan")),
        key_identity=KeyIdentity(team_id=10**20, end_user_id=float("inf"), rejected=("agent_id",)),
        guardrail_info=(GuardrailEntry(mapping=True, modes=("pre_call", None)),),
        outcome=Outcome.success,
    )

    _ = await logger.run_rust(
        Hook.attempt_ended,
        frozenset({Field.cost, Field.key_identity, Field.guardrail_info, Field.outcome}),
        view,
        CallContext(hook=Hook.attempt_ended, call=None, state=None),
    )

    fields: Final = handle.calls[0][1]
    cost: Final = _REQUEST.validate_python(fields["cost"])
    identity: Final = _REQUEST.validate_python(fields["key_identity"])
    response_cost: Final = cost["response_cost"]
    assert isinstance(response_cost, float)
    assert math.isnan(response_cost)
    assert (identity["team_id"], identity["end_user_id"], identity["rejected"]) == (10**20, float("inf"), ("agent_id",))
    assert fields["guardrail_info"] == (
        {"rejected": None, "mapping": True, "modes": ("pre_call", None), "duration": None},
    )
    assert fields["outcome"] == "success"


async def test_a_value_carried_as_its_text_reaches_the_rust_as_text_and_truthiness() -> None:
    handle: Final = Handle()
    logger: Final = rust_logger(handle)
    assert isinstance(logger, RustLogger)
    view: Final = SimpleNamespace(
        key_identity=KeyIdentity(team_id=["t1"], team_priority={}),
        request_tags=(
            Stringified(
                text="[]",
                truthy=False,
                hashable=False,
                integer=False,
                type_name="list",
                hash_error="unhashable type: 'list'",
            ),
            "prod",
        ),
        budget_reservation=BudgetReservation(entries=(), entries_is_list=False),
    )

    _ = await logger.run_rust(
        Hook.pre_call,
        frozenset({Field.key_identity, Field.request_tags, Field.budget_reservation}),
        view,
        CallContext(hook=Hook.pre_call, call=None, state=None),
    )

    fields: Final = handle.calls[0][1]
    identity: Final = _REQUEST.validate_python(fields["key_identity"])
    reservation: Final = _REQUEST.validate_python(fields["budget_reservation"])
    assert (identity["team_id"], identity["team_priority"]) == (
        {
            "text": "['t1']",
            "truthy": True,
            "hashable": False,
            "integer": False,
            "type_name": "list",
            "hash_error": "unhashable type: 'list'",
        },
        {
            "text": "{}",
            "truthy": False,
            "hashable": False,
            "integer": False,
            "type_name": "dict",
            "hash_error": "unhashable type: 'dict'",
        },
    )
    assert fields["request_tags"] == (
        {
            "text": "[]",
            "truthy": False,
            "hashable": False,
            "integer": False,
            "type_name": "list",
            "hash_error": "unhashable type: 'list'",
        },
        "prod",
    )
    assert (reservation["entries"], reservation["entries_is_list"]) == ((), False)


async def test_an_infinite_budget_reservation_reaches_the_rust_as_a_python_float() -> None:
    handle: Final = Handle()
    logger: Final = rust_logger(handle)
    assert isinstance(logger, RustLogger)
    view: Final = SimpleNamespace(
        budget_reservation=BudgetReservation(
            reserved_cost=float("inf"),
            input_cost=float("nan"),
            entries=(ReservationEntry(counter_key="k", reserved_cost=float("-inf")), None),
        )
    )

    _ = await logger.run_rust(
        Hook.pre_call,
        frozenset({Field.budget_reservation}),
        view,
        CallContext(hook=Hook.pre_call, call=None, state=None),
    )

    reservation: Final = _REQUEST.validate_python(handle.calls[0][1]["budget_reservation"])
    input_cost: Final = reservation["input_cost"]
    assert reservation["reserved_cost"] == float("inf")
    assert isinstance(input_cost, float)
    assert math.isnan(input_cost)
    assert reservation["entries"] == (
        {
            "rejected": None,
            "counter_key": "k",
            "entity_type": None,
            "entity_id": None,
            "reserved_cost": float("-inf"),
            "applied_adjustment": None,
        },
        None,
    )
