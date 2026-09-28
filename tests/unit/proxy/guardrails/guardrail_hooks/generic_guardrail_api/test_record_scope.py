from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.exceptions import GuardrailRaisedException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPI
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.record_scope import (
    RecordScope,
    returned_unchanged,
)
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GuardrailInformationScope,
    GuardrailToolParam,
)
from litellm.types.utils import GenericGuardrailAPIInputs

Responder = Callable[[httpx.Request], httpx.Response]

_KEY_A: Final = MappingProxyType({"user_api_key_hash": "hash-a"})
_USER_ROW: Final = MappingProxyType({"role": "user", "content": "hello"})
_TOOL: Final = MappingProxyType(
    {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "required": ["q"]}}}
)


def _chat_request(**extra: object) -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(texts=["hello"], structured_messages=[dict(_USER_ROW)], model="gpt-test", **extra)


def _chat_response() -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(texts=["hi there"], model="gpt-test")


def _respond_with(body: Mapping[str, object]) -> Responder:
    return lambda request: httpx.Response(200, json=dict(body), request=request)


_allow: Final = _respond_with({"action": "NONE"})
_echo_texts: Final = _respond_with({"action": "NONE", "texts": ["hello"]})
_echo_rows: Final = _respond_with({"action": "NONE", "structured_messages": [dict(_USER_ROW)]})
_block: Final = _respond_with({"action": "BLOCKED", "blocked_reason": "policy"})
_mask: Final = _respond_with({"action": "GUARDRAIL_INTERVENED", "texts": ["[REDACTED]"]})
_rewrite_texts: Final = _respond_with({"action": "NONE", "texts": ["[REDACTED]"]})
_rewrite_rows: Final = _respond_with({"action": "NONE", "structured_messages": [{"role": "user", "content": "[x]"}]})
_intervene_without_rewrite: Final = _respond_with({"action": "GUARDRAIL_INTERVENED"})


def _unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _unavailable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, request=request)


def _in_order(*responders: Responder) -> Responder:
    remaining: Final[Iterator[Responder]] = iter(responders)
    return lambda request: next(remaining)(request)


class _Endpoint:
    def __init__(self, respond: Responder) -> None:
        self.requests: Final[list[httpx.Request]] = []  # mutable-ok: records what the endpoint received
        self._respond: Final = respond

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)


def _guardrail(endpoint: _Endpoint, **options: object) -> GenericGuardrailAPI:
    return GenericGuardrailAPI(
        api_base="https://guardrail.test",
        guardrail_name="scoped-guardrail",
        event_hook="pre_call",
        async_handler=AsyncHTTPHandler(transport=httpx.MockTransport(endpoint)),
        **options,
    )


def _turn(
    session_id: str | None = "session-1",
    metadata: Mapping[str, str] = _KEY_A,
) -> dict[str, object]:  # mutable-ok: the decorator appends entries into the request data
    base: Final = {"metadata": dict(metadata)}
    return base if session_id is None else {**base, "litellm_session_id": session_id}


def _recorded(request_data: Mapping[str, object]) -> list[str]:
    metadata: Final = request_data.get("metadata")
    assert isinstance(metadata, dict)
    return [entry["guardrail_status"] for entry in metadata.get("standard_logging_guardrail_information") or []]


async def _run(
    guardrail: GenericGuardrailAPI,
    request_data: dict[str, object],
    input_type: Literal["request", "response"] = "request",
    inputs: GenericGuardrailAPIInputs | None = None,
) -> list[str]:
    await guardrail.apply_guardrail(
        inputs=_chat_request() if inputs is None else inputs, request_data=request_data, input_type=input_type
    )
    return _recorded(request_data)


async def _run_blocked(guardrail: GenericGuardrailAPI, request_data: dict[str, object]) -> list[str]:
    with pytest.raises(GuardrailRaisedException):
        await guardrail.apply_guardrail(inputs=_chat_request(), request_data=request_data, input_type="request")
    return _recorded(request_data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options", [{}, {"guardrail_information_scope": None}, {"guardrail_information_scope": "per_call"}]
)
async def test_per_call_records_every_call_of_a_session(options: dict[str, object]) -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), **options)

    assert [await _run(guardrail, _turn()) for _ in range(3)] == [["success"]] * 3


@pytest.mark.asyncio
async def test_per_session_records_the_first_call_only_but_still_runs_every_call() -> None:
    endpoint: Final = _Endpoint(_allow)
    guardrail: Final = _guardrail(endpoint, guardrail_information_scope="per_session")

    assert [await _run(guardrail, _turn()) for _ in range(3)] == [["success"], [], []]
    assert len(endpoint.requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["per_session", "off"])
@pytest.mark.parametrize(
    ("respond", "inputs"),
    [
        (_allow, _chat_request()),
        (_allow, _chat_request(images=[])),
        (_echo_texts, _chat_request()),
        (_echo_rows, _chat_request()),
        (_allow, _chat_request(tools=[dict(_TOOL)])),
        (_respond_with({"action": "NONE", "tools": [dict(_TOOL)]}), _chat_request(tools=[dict(_TOOL)])),
    ],
    ids=["allow", "empty-images", "echoed-texts", "echoed-rows", "passed-through-tools", "echoed-tools"],
)
async def test_unchanged_chat_allows_are_deduped_after_the_first_turn(
    scope: GuardrailInformationScope, respond: Responder, inputs: GenericGuardrailAPIInputs
) -> None:
    guardrail: Final = _guardrail(_Endpoint(respond), guardrail_information_scope=scope)

    turns: Final = [await _run(guardrail, _turn(), inputs=inputs) for _ in range(2)]

    assert turns[1] == []
    assert turns[0] == (["success"] if scope == "per_session" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize(("scope", "expected"), [("per_session", [["success"], []]), ("off", [[], []])])
async def test_tool_use_only_responses_without_texts_are_deduped(
    scope: GuardrailInformationScope, expected: list[list[str]]
) -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope=scope)
    tool_use_only: Final = GenericGuardrailAPIInputs(
        tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        model="gpt-test",
    )

    assert [await _run(guardrail, _turn(), "response", tool_use_only) for _ in range(2)] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("scope", "expected"), [("per_call", [1, 1]), ("per_session", [1, 0]), ("off", [0, 0])])
async def test_chat_completion_turns_through_the_openai_handler_follow_the_scope(
    scope: GuardrailInformationScope, expected: list[int]
) -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope=scope)

    async def chat_turn() -> int:
        data: Final = {
            "model": "gpt-test",
            "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}],
            "tools": [dict(_TOOL)],
            "metadata": dict(_KEY_A),
            "litellm_session_id": "chat-session",
        }
        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)
        return len(_recorded(data))

    assert [await chat_turn() for _ in range(2)] == expected


@pytest.mark.asyncio
async def test_per_session_records_the_first_call_of_each_session() -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope="per_session")

    assert [await _run(guardrail, _turn(session_id)) for session_id in ("s1", "s1", "s2", "s2")] == [
        ["success"],
        [],
        ["success"],
        [],
    ]


@pytest.mark.asyncio
async def test_per_session_records_the_first_request_and_the_first_response_of_a_session() -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope="per_session")

    async def run_turn() -> list[str]:
        request_data: Final = _turn()
        await _run(guardrail, request_data, "request")
        return await _run(guardrail, request_data, "response", _chat_response())

    first_turn: Final = await run_turn()
    second_turn: Final = await run_turn()

    assert (first_turn, second_turn) == (["success", "success"], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_field", ["user_api_key_hash", "user_api_key_team_id"])
async def test_per_session_dedups_per_authenticated_caller(identity_field: str) -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope="per_session")

    statuses: Final = [
        await _run(guardrail, _turn("shared-id", {identity_field: tenant}))
        for tenant in ("tenant-a", "tenant-b", "tenant-a", "tenant-b")
    ]

    assert statuses == [["success"], ["success"], [], []]


@pytest.mark.asyncio
async def test_per_session_key_hash_takes_precedence_over_team_id() -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope="per_session")
    same_team: Final = (
        {"user_api_key_hash": "hash-a", "user_api_key_team_id": "team"},
        {"user_api_key_hash": "hash-b", "user_api_key_team_id": "team"},
    )

    assert [await _run(guardrail, _turn("shared-id", identity)) for identity in same_team] == [["success"]] * 2


@pytest.mark.asyncio
async def test_per_session_dedups_callers_without_a_key_or_team_apart_from_authenticated_ones() -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope="per_session")

    statuses: Final = [await _run(guardrail, _turn("shared-id", identity)) for identity in ({}, {}, _KEY_A, _KEY_A)]

    assert statuses == [["success"], [], ["success"], []]


@pytest.mark.asyncio
async def test_per_session_without_a_session_id_records_every_call() -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope="per_session")

    assert [await _run(guardrail, _turn(session_id=None)) for _ in range(3)] == [["success"]] * 3


@pytest.mark.asyncio
async def test_per_session_reads_the_session_id_from_metadata() -> None:
    guardrail: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope="per_session")

    statuses: Final = [
        await _run(guardrail, _turn(None, {"user_api_key_hash": "hash-a", "session_id": "meta-session"}))
        for _ in range(2)
    ]

    assert statuses == [["success"], []]


@pytest.mark.asyncio
async def test_off_still_runs_every_call() -> None:
    endpoint: Final = _Endpoint(_allow)
    guardrail: Final = _guardrail(endpoint, guardrail_information_scope="off")

    assert [await _run(guardrail, _turn(session_id)) for session_id in ("s1", "s1", None)] == [[], [], []]
    assert len(endpoint.requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["per_session", "off"])
@pytest.mark.parametrize(
    ("respond", "inputs"),
    [
        (_mask, _chat_request()),
        (_rewrite_texts, _chat_request()),
        (_rewrite_rows, _chat_request()),
        (_intervene_without_rewrite, _chat_request()),
        (_respond_with({"action": "NONE", "images": ["data:image/png;base64,Yg=="]}), _chat_request(images=["a"])),
        (
            _respond_with({"action": "NONE", "tools": [{"type": "function", "function": {"name": "other"}}]}),
            _chat_request(tools=[dict(_TOOL)]),
        ),
    ],
    ids=[
        "mask",
        "rewritten-texts",
        "rewritten-rows",
        "intervention-without-rewrite",
        "rewritten-images",
        "rewritten-tools",
    ],
)
async def test_rewrites_and_interventions_are_recorded_under_every_scope(
    scope: GuardrailInformationScope, respond: Responder, inputs: GenericGuardrailAPIInputs
) -> None:
    guardrail: Final = _guardrail(_Endpoint(respond), guardrail_information_scope=scope)

    assert [await _run(guardrail, _turn(), inputs=inputs) for _ in range(2)] == [["success"]] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["per_session", "off"])
async def test_blocks_are_recorded_under_every_scope(scope: GuardrailInformationScope) -> None:
    guardrail: Final = _guardrail(_Endpoint(_block), guardrail_information_scope=scope)

    assert [await _run_blocked(guardrail, _turn()) for _ in range(2)] == [["guardrail_intervened"]] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["per_session", "off"])
async def test_endpoint_failures_are_recorded_under_every_scope(scope: GuardrailInformationScope) -> None:
    guardrail: Final = _guardrail(_Endpoint(_unreachable), guardrail_information_scope=scope)
    request_data: Final = _turn()

    with pytest.raises(Exception, match="Generic Guardrail API failed"):
        await guardrail.apply_guardrail(inputs=_chat_request(), request_data=request_data, input_type="request")

    assert _recorded(request_data) == ["guardrail_failed_to_respond"]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["per_session", "off"])
@pytest.mark.parametrize("respond", [_unreachable, _unavailable])
async def test_fail_open_passthroughs_are_recorded_under_every_scope(
    scope: GuardrailInformationScope, respond: Responder
) -> None:
    guardrail: Final = _guardrail(
        _Endpoint(respond), guardrail_information_scope=scope, unreachable_fallback="fail_open"
    )

    assert [await _run(guardrail, _turn()) for _ in range(2)] == [["success"]] * 2


@pytest.mark.asyncio
async def test_a_block_does_not_claim_the_session() -> None:
    guardrail: Final = _guardrail(
        _Endpoint(_in_order(_block, _allow, _allow)), guardrail_information_scope="per_session"
    )

    blocked: Final = await _run_blocked(guardrail, _turn())
    allowed: Final = [await _run(guardrail, _turn()) for _ in range(2)]

    assert (blocked, allowed) == (["guardrail_intervened"], [["success"], []])


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [_mask, _intervene_without_rewrite])
async def test_a_rewrite_or_intervention_does_not_claim_the_session(first: Responder) -> None:
    guardrail: Final = _guardrail(
        _Endpoint(_in_order(first, _allow, _allow)), guardrail_information_scope="per_session"
    )

    assert [await _run(guardrail, _turn()) for _ in range(3)] == [["success"], ["success"], []]


@pytest.mark.asyncio
async def test_a_fail_open_passthrough_does_not_claim_the_session() -> None:
    guardrail: Final = _guardrail(
        _Endpoint(_in_order(_unavailable, _allow, _allow)),
        guardrail_information_scope="per_session",
        unreachable_fallback="fail_open",
    )

    assert [await _run(guardrail, _turn()) for _ in range(3)] == [["success"], ["success"], []]


@pytest.mark.asyncio
async def test_each_guardrail_instance_records_its_own_first_call_of_a_session() -> None:
    first: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope="per_session")
    second: Final = _guardrail(_Endpoint(_allow), guardrail_information_scope="per_session")

    statuses: Final = [await _run(guardrail, _turn("shared-id")) for guardrail in (first, second, first, second)]

    assert statuses == [["success"], ["success"], [], []]


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
        return record_scope.should_record_allow(session_id="s1", tenant="hash-a", input_type="request")

    first, within_ttl = record(), record()
    now[0] = 61.0

    assert (first, within_ttl, record()) == (True, False, True)


def test_undecodable_bytes_compare_without_raising() -> None:
    rows: Final = [{"role": "user", "content": b"\xff\xfe"}]

    assert returned_unchanged(
        GenericGuardrailAPIInputs(texts=["hello"], structured_messages=rows),
        GenericGuardrailAPIInputs(texts=["hello"], structured_messages=rows),
    )
