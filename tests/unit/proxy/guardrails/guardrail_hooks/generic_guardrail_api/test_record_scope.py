from types import MappingProxyType
from typing import Final

import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.record_scope import (
    Caller,
    RecordScope,
    returned_unchanged,
)
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GuardrailInformationScope,
    GuardrailToolParam,
)
from litellm.types.utils import GenericGuardrailAPIInputs

_USER_ROW: Final = MappingProxyType({"role": "user", "content": "hello"})
_TOOL: Final = MappingProxyType(
    {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "required": ["q"]}}}
)


def _chat_request(**extra: object) -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(texts=["hello"], structured_messages=[dict(_USER_ROW)], model="gpt-test", **extra)


@pytest.mark.parametrize(
    ("returned_tools", "unchanged"),
    [
        ([GuardrailToolParam.model_validate(dict(_TOOL))], True),
        ([GuardrailToolParam.model_validate({"type": "function", "function": {"name": "other"}})], False),
    ],
)
def test_tools_returned_as_models_compare_by_content(returned_tools: list[GuardrailToolParam], unchanged: bool) -> None:
    tool_with_a_tuple: Final = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "required": ("q",)}},
    }
    sent: Final = _chat_request(tools=[tool_with_a_tuple])
    returned: Final = GenericGuardrailAPIInputs(texts=["hello"], tools=returned_tools)

    assert returned_unchanged(sent, returned) is unchanged


def test_per_session_records_again_once_the_session_has_expired() -> None:
    now: Final = [0.0]  # mutable-ok: fake clock advanced by the test
    record_scope: Final = RecordScope(
        "per_session", recorded_sessions=InMemoryCache(default_ttl=60, clock=lambda: now[0])
    )

    def record() -> bool:
        return record_scope.should_record_allow(
            session_id="s1", caller=Caller(key_hash="hash-a", team_id=None, user_id=None), input_type="request"
        )

    first, within_ttl = record(), record()
    now[0] = 61.0

    assert (first, within_ttl, record()) == (True, False, True)


def test_undecodable_bytes_compare_without_raising() -> None:
    rows: Final = [{"role": "user", "content": b"\xff\xfe"}]

    assert returned_unchanged(
        GenericGuardrailAPIInputs(texts=["hello"], structured_messages=rows),
        GenericGuardrailAPIInputs(texts=["hello"], structured_messages=rows),
    )


class _Opaque:
    pass


def test_a_non_json_value_passed_through_compares_as_unchanged() -> None:
    passed_through: Final = _Opaque()

    assert returned_unchanged(
        _chat_request(tools=[passed_through]), GenericGuardrailAPIInputs(texts=["hello"], tools=[passed_through])
    )


@pytest.mark.parametrize(("scope", "recorded"), [("per_call", True), ("off", False)])
def test_per_call_and_off_decide_without_claiming_a_session(scope: GuardrailInformationScope, recorded: bool) -> None:
    sessions: Final = InMemoryCache()
    record_scope: Final = RecordScope(scope, recorded_sessions=sessions)
    caller: Final = Caller(key_hash="hash-a", team_id=None, user_id=None)

    decisions: Final = [
        record_scope.should_record_allow(session_id="s1", caller=caller, input_type="request") for _ in range(2)
    ]

    session_still_unclaimed: Final = RecordScope("per_session", recorded_sessions=sessions).should_record_allow(
        session_id="s1", caller=caller, input_type="request"
    )

    assert (decisions, session_still_unclaimed) == ([recorded, recorded], True)
