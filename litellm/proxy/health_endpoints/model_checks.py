"""Live checks behind ``POST /health/test_model``.

One three-turn tool-call conversation per selected endpoint, read from that endpoint's file in
``test_model_requests/``. ``turn_1`` is the full first request body in the endpoint's native shape. Chat Completions
and Responses carry no prompt-cache markers, so later turns can only report cached tokens when the provider caches
the repeated prefix on its own. Messages mirrors Claude Code: ``cache_control`` on the system prompt and a
mid-conversation ``role: system`` reminder, so a proxy that hoists that reminder into the system prompt shows up as a
cache miss. ``turn_2_appended_messages`` lists what the second request adds after turn 1's messages (the model's tool
call echoed back plus the tool result), ``turn_3_appended_messages`` what the third adds after that (the model's
answer echoed back, the reminder, a user follow-up), with ``<turn 1 response: assistant output>``,
``<turn 1 response: tool call id>`` and ``<turn 2 response: assistant output>`` filled in from the model's replies.
Three verdicts come out of the three turns: the conversation itself (every turn parses, the tool call lands, the
tool result is used, the follow-up is answered from the earlier turn), prompt caching, and cost tracking.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict

TestModelEndpoint: TypeAlias = Literal["/v1/chat/completions", "/v1/responses", "/v1/messages"]
TestModelRouteType: TypeAlias = Literal["acompletion", "aresponses", "anthropic_messages"]
Verdict: TypeAlias = Literal["pass", "warn", "fail"]
EndpointStatus: TypeAlias = Literal["success", "warning", "error"]

TEST_MODEL_ENDPOINTS: Final[tuple[TestModelEndpoint, ...]] = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
WEATHER_TOOL_NAME: Final = "get_weather"
WEATHER_TOOL_RESULT: Final = "72F and sunny"
TEST_MODEL_REQUESTS_DIR: Final = Path(__file__).resolve().parent / "test_model_requests"
TURN_ONE_ASSISTANT_OUTPUT: Final = "<turn 1 response: assistant output>"
TURN_ONE_TOOL_CALL_ID: Final = "<turn 1 response: tool call id>"
TURN_TWO_ASSISTANT_OUTPUT: Final = "<turn 2 response: assistant output>"


@dataclass(frozen=True, slots=True)
class ToolCallSeen:
    id: str
    name: str
    raw_arguments: str
    arguments: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class ParsedTurn:
    tool_call: ToolCallSeen | None
    text: str
    input_tokens: int
    cache_read_tokens: int
    assistant_echo: object


@dataclass(frozen=True, slots=True)
class EndpointScript:
    endpoint: TestModelEndpoint
    route_type: TestModelRouteType
    metadata_key: Literal["metadata", "litellm_metadata"]
    turn_one: Callable[[], Mapping[str, object]]
    turn_two: Callable[[ParsedTurn], Mapping[str, object]]
    turn_three: Callable[[ParsedTurn, ParsedTurn], Mapping[str, object]]
    parse: Callable[[object], ParsedTurn | str]


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: Literal["conversation", "prompt_caching", "cost_tracking"]
    status: Verdict
    detail: str


@dataclass(frozen=True, slots=True)
class EndpointResult:
    endpoint: TestModelEndpoint
    status: EndpointStatus
    checks: tuple[CheckResult, ...]
    cache_read_tokens: int
    cache_read_percent: float | None
    spend_usd: float
    duration_ms: int


@dataclass(frozen=True, slots=True)
class CalledTurn:
    """One turn's provider response and the cost the proxy logged for it (what the spend log row carries)."""

    response: object
    cost: float | None


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    response: object | None
    cost: float | None
    error: str | None
    call_id: str


@dataclass(frozen=True, slots=True)
class RepeatedTurn:
    """A turn whose prompt repeats the earlier turns, sent once more when the provider reports no cache hit."""

    number: Literal[2, 3]
    outcome: TurnOutcome | None
    parsed: ParsedTurn | str | None
    attempts: int


TurnCaller: TypeAlias = Callable[[EndpointScript, Mapping[str, object], str], Awaitable[CalledTurn]]
SpendLookup: TypeAlias = Callable[[Sequence[str]], Awaitable[Mapping[str, float]]]
CallIdFactory: TypeAlias = Callable[[], str]


class _Loose(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _ChatFunction(_Loose):
    name: str
    arguments: str = ""


class _ChatToolCall(_Loose):
    id: str = ""
    function: _ChatFunction


class _PromptTokensDetails(_Loose):
    cached_tokens: int | None = None


class _ChatUsage(_Loose):
    prompt_tokens: int = 0
    prompt_tokens_details: _PromptTokensDetails | None = None


class _ChatMessage(_Loose):
    content: str | Sequence[object] | None = None
    tool_calls: Sequence[_ChatToolCall] | None = None


class _ChatChoice(_Loose):
    message: _ChatMessage


class _ChatResponse(_Loose):
    choices: Sequence[_ChatChoice]
    usage: _ChatUsage | None = None


class _ResponsesContent(_Loose):
    type: str = ""
    text: str = ""


class _ResponsesItem(_Loose):
    type: str = ""
    name: str = ""
    arguments: str = ""
    call_id: str = ""
    content: Sequence[_ResponsesContent] | None = None


class _InputTokensDetails(_Loose):
    cached_tokens: int | None = None


class _ResponsesUsage(_Loose):
    input_tokens: int = 0
    input_tokens_details: _InputTokensDetails | None = None


class _ResponsesResponse(_Loose):
    output: Sequence[_ResponsesItem]
    usage: _ResponsesUsage | None = None


class _MessagesBlock(_Loose):
    type: str = ""
    text: str = ""
    id: str = ""
    name: str = ""
    input: Mapping[str, object] | None = None


class _MessagesUsage(_Loose):
    input_tokens: int = 0
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class _MessagesResponse(_Loose):
    content: Sequence[_MessagesBlock]
    usage: _MessagesUsage


_CHAT: Final = TypeAdapter(_ChatResponse)
_RESPONSES: Final = TypeAdapter(_ResponsesResponse)
_MESSAGES: Final = TypeAdapter(_MessagesResponse)
_PLAIN_OBJECT: Final = TypeAdapter(dict[str, object])
_OBJECT_LIST: Final = TypeAdapter(list[object])
_CHOICES: Final = TypeAdapter(list[dict[str, object]])
_JSON_DOCUMENT: Final = TypeAdapter(object)
_EMPTY_ASSISTANT: Final[Mapping[str, object]] = MappingProxyType({"role": "assistant", "content": None})


class RequestFile(BaseModel):
    """One three-turn conversation on disk: ``turn_1`` is the full first request body, ``turn_2_appended_messages``
    the messages the second request adds after it, ``turn_3_appended_messages`` the ones the third adds after those,
    with the placeholders filled from the earlier responses."""

    model_config = ConfigDict(frozen=True)

    turn_1: Mapping[str, object]
    turn_2_appended_messages: Sequence[object]
    turn_3_appended_messages: Sequence[object]


def _request_file(name: str) -> Callable[[], RequestFile]:
    text: Final = (TEST_MODEL_REQUESTS_DIR / f"{name}.json").read_text(encoding="utf-8")
    return lambda: RequestFile.model_validate_json(text)


_CHAT_FILE: Final = _request_file("chat_completions")
_RESPONSES_FILE: Final = _request_file("responses")
_MESSAGES_FILE: Final = _request_file("messages")


Placeholders: TypeAlias = Mapping[str, object]
ConversationKey: TypeAlias = Literal["messages", "input"]


def _replace_placeholder(text: str, placeholder: tuple[str, object]) -> str:
    return text.replace(json.dumps(placeholder[0]), json.dumps(placeholder[1]))


def _fill(item: object, placeholders: Placeholders) -> object:
    return _JSON_DOCUMENT.validate_json(reduce(_replace_placeholder, placeholders.items(), json.dumps(item)))


def _appended_messages(item: object, placeholders: Placeholders) -> Sequence[object]:
    echo: Final = placeholders.get(item) if isinstance(item, str) else None
    if isinstance(echo, list):
        return _OBJECT_LIST.validate_python(echo)
    return (_fill(item, placeholders),)


def _extended(
    body: Mapping[str, object], key: ConversationKey, items: Sequence[object], placeholders: Placeholders
) -> Mapping[str, object]:
    appended: Final = chain.from_iterable(_appended_messages(item, placeholders) for item in items)
    return {**body, key: [*_message_list(body.get(key)), *appended]}


def _turn_one_placeholders(turn_one: ParsedTurn) -> Placeholders:
    return MappingProxyType(
        {
            TURN_ONE_TOOL_CALL_ID: turn_one.tool_call.id if turn_one.tool_call is not None else "",
            TURN_ONE_ASSISTANT_OUTPUT: turn_one.assistant_echo,
        }
    )


def _turn_one(file: Callable[[], RequestFile]) -> Callable[[], Mapping[str, object]]:
    return lambda: file().turn_1


def _turn_two(file: Callable[[], RequestFile], key: ConversationKey) -> Callable[[ParsedTurn], Mapping[str, object]]:
    def build(turn_one: ParsedTurn) -> Mapping[str, object]:
        first: Final = file()
        return _extended(first.turn_1, key, first.turn_2_appended_messages, _turn_one_placeholders(turn_one))

    return build


def _turn_three(
    file: Callable[[], RequestFile], key: ConversationKey
) -> Callable[[ParsedTurn, ParsedTurn], Mapping[str, object]]:
    def build(turn_one: ParsedTurn, turn_two: ParsedTurn) -> Mapping[str, object]:
        first: Final = file()
        second: Final = _turn_two(file, key)(turn_one)
        return _extended(
            second,
            key,
            first.turn_3_appended_messages,
            MappingProxyType({TURN_TWO_ASSISTANT_OUTPUT: turn_two.assistant_echo}),
        )

    return build


def _plain(response: object) -> Mapping[str, object] | None:
    if isinstance(response, BaseModel):
        return _PLAIN_OBJECT.validate_python(response.model_dump(exclude_none=True))
    if isinstance(response, dict):
        return _PLAIN_OBJECT.validate_python(response)
    return None


def _validation_detail(error: ValidationError) -> str:
    first: Final = error.errors()[0]
    location: Final = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}" if location else first["msg"]


def _arguments(raw: str) -> Mapping[str, object] | None:
    try:
        parsed: Final = _JSON_DOCUMENT.validate_json(raw)
    except ValidationError:
        return None
    return MappingProxyType(_PLAIN_OBJECT.validate_python(parsed)) if isinstance(parsed, dict) else None


def _message_list(messages: object) -> Sequence[object]:
    return _OBJECT_LIST.validate_python(messages) if isinstance(messages, list) else []


def _first_choice_message(plain: Mapping[str, object]) -> object:
    try:
        choices: Final = _CHOICES.validate_python(plain.get("choices"))
    except ValidationError:
        return dict(_EMPTY_ASSISTANT)
    return choices[0].get("message", dict(_EMPTY_ASSISTANT)) if choices else dict(_EMPTY_ASSISTANT)


def _parse_chat(response: object) -> ParsedTurn | str:
    plain: Final = _plain(response)
    if plain is None:
        return f"response is a {type(response).__name__}, not a chat completion"
    try:
        parsed: Final = _CHAT.validate_python(plain)
    except ValidationError as error:
        return f"chat completion did not parse: {_validation_detail(error)}"
    if not parsed.choices:
        return "chat completion returned no choices"
    message: Final = parsed.choices[0].message
    first_call: Final = message.tool_calls[0] if message.tool_calls else None
    usage: Final = parsed.usage
    details: Final = usage.prompt_tokens_details if usage is not None else None
    return ParsedTurn(
        tool_call=(
            ToolCallSeen(
                id=first_call.id,
                name=first_call.function.name,
                raw_arguments=first_call.function.arguments,
                arguments=_arguments(first_call.function.arguments),
            )
            if first_call is not None
            else None
        ),
        text=message.content if isinstance(message.content, str) else "",
        input_tokens=usage.prompt_tokens if usage is not None else 0,
        cache_read_tokens=(details.cached_tokens or 0) if details is not None else 0,
        assistant_echo=_first_choice_message(plain),
    )


def _parse_responses(response: object) -> ParsedTurn | str:
    plain: Final = _plain(response)
    if plain is None:
        return f"response is a {type(response).__name__}, not a Responses API response"
    try:
        parsed: Final = _RESPONSES.validate_python(plain)
    except ValidationError as error:
        return f"Responses API response did not parse: {_validation_detail(error)}"
    function_call: Final = next((item for item in parsed.output if item.type == "function_call"), None)
    texts: Final = tuple(
        content.text
        for item in parsed.output
        if item.type == "message" and item.content is not None
        for content in item.content  # comprehension-ok: flattening message content blocks
        if content.type == "output_text"
    )
    usage: Final = parsed.usage
    details: Final = usage.input_tokens_details if usage is not None else None
    return ParsedTurn(
        tool_call=(
            ToolCallSeen(
                id=function_call.call_id,
                name=function_call.name,
                raw_arguments=function_call.arguments,
                arguments=_arguments(function_call.arguments),
            )
            if function_call is not None
            else None
        ),
        text=" ".join(texts),
        input_tokens=usage.input_tokens if usage is not None else 0,
        cache_read_tokens=(details.cached_tokens or 0) if details is not None else 0,
        assistant_echo=plain.get("output", []),
    )


def _parse_messages(response: object) -> ParsedTurn | str:
    plain: Final = _plain(response)
    if plain is None:
        return f"response is a {type(response).__name__}, not a Messages API response"
    try:
        parsed: Final = _MESSAGES.validate_python(plain)
    except ValidationError as error:
        return f"Messages API response did not parse: {_validation_detail(error)}"
    tool_use: Final = next((block for block in parsed.content if block.type == "tool_use"), None)
    cache_read: Final = parsed.usage.cache_read_input_tokens or 0
    cache_creation: Final = parsed.usage.cache_creation_input_tokens or 0
    return ParsedTurn(
        tool_call=(
            ToolCallSeen(
                id=tool_use.id,
                name=tool_use.name,
                raw_arguments=json.dumps(tool_use.input or {}),
                arguments=MappingProxyType(dict(tool_use.input or {})),
            )
            if tool_use is not None
            else None
        ),
        text=" ".join(block.text for block in parsed.content if block.type == "text"),
        input_tokens=parsed.usage.input_tokens + cache_read + cache_creation,
        cache_read_tokens=cache_read,
        assistant_echo=plain.get("content", []),
    )


CHAT_COMPLETIONS_SCRIPT: Final = EndpointScript(
    endpoint="/v1/chat/completions",
    route_type="acompletion",
    metadata_key="metadata",
    turn_one=_turn_one(_CHAT_FILE),
    turn_two=_turn_two(_CHAT_FILE, "messages"),
    turn_three=_turn_three(_CHAT_FILE, "messages"),
    parse=_parse_chat,
)
RESPONSES_SCRIPT: Final = EndpointScript(
    endpoint="/v1/responses",
    route_type="aresponses",
    metadata_key="litellm_metadata",
    turn_one=_turn_one(_RESPONSES_FILE),
    turn_two=_turn_two(_RESPONSES_FILE, "input"),
    turn_three=_turn_three(_RESPONSES_FILE, "input"),
    parse=_parse_responses,
)
MESSAGES_SCRIPT: Final = EndpointScript(
    endpoint="/v1/messages",
    route_type="anthropic_messages",
    metadata_key="litellm_metadata",
    turn_one=_turn_one(_MESSAGES_FILE),
    turn_two=_turn_two(_MESSAGES_FILE, "messages"),
    turn_three=_turn_three(_MESSAGES_FILE, "messages"),
    parse=_parse_messages,
)
SCRIPTS: Final[Mapping[TestModelEndpoint, EndpointScript]] = MappingProxyType(
    {script.endpoint: script for script in (CHAT_COMPLETIONS_SCRIPT, RESPONSES_SCRIPT, MESSAGES_SCRIPT)}
)
ENDPOINT_FOR_MODE: Final[Mapping[str, TestModelEndpoint]] = MappingProxyType(
    {"chat": "/v1/chat/completions", "responses": "/v1/responses", "anthropic_messages": "/v1/messages"}
)


def response_cost(response: object) -> float | None:
    cost: Final = get_hidden_params_dict(response).get("response_cost")
    return float(cost) if isinstance(cost, int | float) else None


async def _turn(script: EndpointScript, body: Mapping[str, object], call: TurnCaller, call_id: str) -> TurnOutcome:
    try:
        called: Final = await call(script, body, call_id)
    except Exception as error:  # noqa: BLE001  # a provider or translation failure is the check's finding
        return TurnOutcome(response=None, cost=None, error=f"{type(error).__name__}: {error}", call_id=call_id)
    return TurnOutcome(response=called.response, cost=called.cost, error=None, call_id=call_id)


def _answer_failure(turn: RepeatedTurn, purpose: str, expectation: str) -> CheckResult | None:
    if turn.outcome is None or turn.outcome.error is not None:
        reason: Final = turn.outcome.error if turn.outcome is not None else "not sent"
        return CheckResult("conversation", "fail", f"turn {turn.number} request with {purpose} failed: {reason}")
    if isinstance(turn.parsed, str):
        return CheckResult("conversation", "fail", f"turn {turn.number} {turn.parsed}")
    if turn.parsed is None or "72" not in turn.parsed.text:
        answer: Final = turn.parsed.text[:160] if turn.parsed is not None else ""
        return CheckResult("conversation", "fail", f"turn {turn.number} {expectation}: {answer!r}")
    return None


def _conversation_check(
    turn_one: TurnOutcome,
    parsed_one: ParsedTurn | str | None,
    second: RepeatedTurn,
    third: RepeatedTurn,
) -> CheckResult:
    if turn_one.error is not None:
        return CheckResult("conversation", "fail", f"turn 1 request failed: {turn_one.error}")
    if isinstance(parsed_one, str):
        return CheckResult("conversation", "fail", f"turn 1 {parsed_one}")
    if parsed_one is None or parsed_one.tool_call is None:
        shown: Final = parsed_one.text[:160] if parsed_one is not None else ""
        return CheckResult("conversation", "fail", f"turn 1 returned no tool call, the model answered: {shown!r}")
    call: Final = parsed_one.tool_call
    if call.name != WEATHER_TOOL_NAME:
        return CheckResult("conversation", "fail", f"turn 1 called {call.name!r} instead of {WEATHER_TOOL_NAME!r}")
    if call.arguments is None:
        return CheckResult(
            "conversation", "fail", f"turn 1 tool arguments are not a JSON object: {call.raw_arguments!r}"
        )
    if "location" not in call.arguments:
        return CheckResult("conversation", "fail", f"turn 1 tool arguments lack 'location': {call.raw_arguments!r}")
    second_failure: Final = _answer_failure(
        second, "the tool result", f"did not use the tool result {WEATHER_TOOL_RESULT!r}"
    )
    if second_failure is not None:
        return second_failure
    third_failure: Final = _answer_failure(
        third, "the follow-up question", "did not carry the temperature from turn 2 into the follow-up answer"
    )
    if third_failure is not None:
        return third_failure
    answers: Final = ", ".join(
        f"turn {turn.number} answered {turn.parsed.text[:120]!r}"
        for turn in (second, third)
        if isinstance(turn.parsed, ParsedTurn)
    )
    return CheckResult("conversation", "pass", f"turn 1 called {WEATHER_TOOL_NAME}({call.raw_arguments}), {answers}")


def _cache_percent(parsed: ParsedTurn | None) -> float | None:
    if parsed is None or parsed.input_tokens <= 0:
        return None
    return round(100.0 * parsed.cache_read_tokens / parsed.input_tokens, 1)


def _missed_cache(parsed: ParsedTurn | str | None) -> bool:
    return isinstance(parsed, ParsedTurn) and parsed.cache_read_tokens == 0


def _hit_cache(parsed: ParsedTurn | str | None) -> bool:
    return isinstance(parsed, ParsedTurn) and parsed.cache_read_tokens > 0


def _parsed(turn: RepeatedTurn) -> ParsedTurn | None:
    return turn.parsed if isinstance(turn.parsed, ParsedTurn) else None


def _cache_read_note(turn: RepeatedTurn, parsed: ParsedTurn) -> str:
    attempt_note: Final = " on the second attempt" if turn.attempts > 1 else ""
    return (
        f"turn {turn.number} read {parsed.cache_read_tokens} of {parsed.input_tokens} input tokens from cache "
        f"({_cache_percent(parsed)}%){attempt_note}"
    )


def _cache_miss(turn: RepeatedTurn, parsed: ParsedTurn) -> CheckResult:
    return CheckResult(
        "prompt_caching",
        "fail",
        f"turn {turn.number} reported 0 cached tokens on {parsed.input_tokens} input tokens in {turn.attempts} "
        "attempts, the provider did not reuse the repeated system prompt",
    )


def _prompt_caching_check(second: RepeatedTurn, third: RepeatedTurn) -> CheckResult:
    reached: Final = tuple((turn, parsed) for turn in (second, third) if (parsed := _parsed(turn)) is not None)
    if not reached:
        return CheckResult("prompt_caching", "fail", "not checked, the conversation did not reach turn 2")
    missed: Final = next((_cache_miss(turn, parsed) for turn, parsed in reached if parsed.cache_read_tokens == 0), None)
    if missed is not None:
        return missed
    return CheckResult("prompt_caching", "pass", ", ".join(_cache_read_note(turn, parsed) for turn, parsed in reached))


def _cost_verdict(turn_number: int, cost: float | None) -> CheckResult | None:
    if cost is None:
        return CheckResult(
            "cost_tracking",
            "fail",
            f"turn {turn_number} carried no response_cost, no price resolved for this model (not in the cost map and "
            "no custom pricing on the deployment)",
        )
    if cost == 0:
        return CheckResult(
            "cost_tracking",
            "warn",
            f"turn {turn_number} response_cost is $0, either this model is priced at zero or no price resolved for "
            "it (not in the cost map and no custom pricing on the deployment)",
        )
    return None


def _cost_tracking_check(turns: Sequence[TurnOutcome], spend_rows: Mapping[str, float] | None) -> CheckResult:
    if not turns:
        return CheckResult("cost_tracking", "fail", "not checked, no turn returned a response")
    costs: Final = tuple(turn.cost for turn in turns)
    failed: Final = next(
        (verdict for number, cost in enumerate(costs, start=1) if (verdict := _cost_verdict(number, cost)) is not None),
        None,
    )
    if failed is not None:
        return failed
    priced: Final = ", ".join(f"turn {number} ${cost:.6f}" for number, cost in enumerate(costs, start=1))
    if spend_rows is None:
        return CheckResult("cost_tracking", "pass", f"{priced}, spend log not checked (no database)")
    missing: Final = tuple(turn.call_id for turn in turns if turn.call_id not in spend_rows)
    if missing:
        return CheckResult(
            "cost_tracking",
            "fail",
            f"{priced}, but no LiteLLM_SpendLogs row landed for request_id {', '.join(missing)}",
        )
    logged: Final = ", ".join(f"${spend_rows[turn.call_id]:.6f}" for turn in turns)
    return CheckResult("cost_tracking", "pass", f"{priced}, spend log rows written ({logged})")


def _status(checks: Sequence[CheckResult]) -> EndpointStatus:
    statuses: Final = frozenset(check.status for check in checks)
    if "fail" in statuses:
        return "error"
    return "warning" if "warn" in statuses else "success"


def overall_status(results: Sequence[EndpointResult]) -> EndpointStatus:
    statuses: Final = frozenset(result.status for result in results)
    if "error" in statuses:
        return "error"
    return "warning" if "warning" in statuses else "success"


async def _repeated_turn(
    script: EndpointScript,
    number: Literal[2, 3],
    build: Callable[[], Mapping[str, object]] | None,
    call: TurnCaller,
    new_call_id: CallIdFactory,
) -> RepeatedTurn:
    if build is None:
        return RepeatedTurn(number=number, outcome=None, parsed=None, attempts=0)
    first: Final = await _turn(script, build(), call, new_call_id())
    first_parsed: Final = script.parse(first.response) if first.response is not None else None
    if not _missed_cache(first_parsed):
        return RepeatedTurn(number=number, outcome=first, parsed=first_parsed, attempts=1)
    retry: Final = await _turn(script, build(), call, new_call_id())
    retry_parsed: Final = script.parse(retry.response) if retry.response is not None else None
    if _hit_cache(retry_parsed):
        return RepeatedTurn(number=number, outcome=retry, parsed=retry_parsed, attempts=2)
    return RepeatedTurn(number=number, outcome=first, parsed=first_parsed, attempts=2)


async def run_endpoint_check(
    script: EndpointScript,
    call: TurnCaller,
    *,
    new_call_id: CallIdFactory,
    spend_lookup: SpendLookup | None,
) -> EndpointResult:
    started: Final = time.monotonic()
    turn_one: Final = await _turn(script, script.turn_one(), call, new_call_id())
    parsed_one: Final = script.parse(turn_one.response) if turn_one.response is not None else None
    second: Final = await _repeated_turn(
        script,
        2,
        partial(script.turn_two, parsed_one)
        if isinstance(parsed_one, ParsedTurn) and parsed_one.tool_call is not None
        else None,
        call,
        new_call_id,
    )
    parsed_two: Final = _parsed(second)
    third: Final = await _repeated_turn(
        script,
        3,
        partial(script.turn_three, parsed_one, parsed_two)
        if isinstance(parsed_one, ParsedTurn) and parsed_two is not None
        else None,
        call,
        new_call_id,
    )
    answered: Final = tuple(
        turn for turn in (turn_one, second.outcome, third.outcome) if turn is not None and turn.response is not None
    )
    spend_rows: Final = (
        await spend_lookup(tuple(turn.call_id for turn in answered)) if spend_lookup is not None and answered else None
    )
    checks: Final = (
        _conversation_check(turn_one, parsed_one, second, third),
        _prompt_caching_check(second, third),
        _cost_tracking_check(answered, spend_rows),
    )
    return EndpointResult(
        endpoint=script.endpoint,
        status=_status(checks),
        checks=checks,
        cache_read_tokens=parsed_two.cache_read_tokens if parsed_two is not None else 0,
        cache_read_percent=_cache_percent(parsed_two),
        spend_usd=sum(turn.cost for turn in answered if turn.cost is not None),
        duration_ms=int((time.monotonic() - started) * 1000),
    )


class TestModelCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["conversation", "prompt_caching", "cost_tracking"]
    status: Verdict
    detail: str


class TestModelEndpointResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    endpoint: TestModelEndpoint
    status: EndpointStatus
    checks: Sequence[TestModelCheck]
    cache_read_tokens: int
    cache_read_percent: float | None
    spend_usd: float
    duration_ms: int


class TestModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: EndpointStatus
    model: str | None
    model_id: str | None
    endpoints: Sequence[TestModelEndpointResult]


def endpoint_result_payload(result: EndpointResult) -> TestModelEndpointResult:
    return TestModelEndpointResult(
        endpoint=result.endpoint,
        status=result.status,
        checks=[TestModelCheck(name=check.name, status=check.status, detail=check.detail) for check in result.checks],
        cache_read_tokens=result.cache_read_tokens,
        cache_read_percent=result.cache_read_percent,
        spend_usd=result.spend_usd,
        duration_ms=result.duration_ms,
    )
