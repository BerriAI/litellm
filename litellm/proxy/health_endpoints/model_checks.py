"""Live checks behind ``POST /health/test_model``.

One two-turn tool-call conversation per selected endpoint, scripted in that endpoint's native request shape
with no prompt-cache markers, so the second turn can only report cached tokens when the provider caches the
repeated prefix on its own. Three verdicts come out of the two turns: the conversation itself (both turns
parse, the tool call lands, the tool result is used), prompt caching, and cost tracking.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict

TestModelEndpoint: TypeAlias = Literal["/v1/chat/completions", "/v1/responses", "/v1/messages"]
TestModelRouteType: TypeAlias = Literal["acompletion", "aresponses", "anthropic_messages"]
Verdict: TypeAlias = Literal["pass", "warn", "fail"]
EndpointStatus: TypeAlias = Literal["success", "warning", "error"]

TEST_MODEL_ENDPOINTS: Final[tuple[TestModelEndpoint, ...]] = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
WEATHER_TOOL_NAME: Final = "get_weather"
WEATHER_TOOL_RESULT: Final = "72F and sunny"
WEATHER_QUESTION: Final = "What is the weather in Boston right now? Use the get_weather tool to find out."
MAX_OUTPUT_TOKENS: Final = 2048
THINKING_BUDGET_TOKENS: Final = 1024

_WEATHER_PARAMETERS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, for example Boston"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    }
)
_WEATHER_DESCRIPTION: Final = "Get the current weather for a city"

_CITIES: Final = ("Boston", "Denver", "Lisbon", "Osaka", "Nairobi", "Perth", "Quito", "Reykjavik")
_SECTION_SENTENCES: Final = (
    "Section {n}: deliveries routed through the {city} hub are scanned at intake, weighed on the calibrated "
    "scale, and matched against the manifest before any label is printed or any pallet leaves the dock.",
    "Agents handling {city} tickets confirm the tracking number, the declared value, and the delivery window "
    "with the sender before promising a reroute, and record the promise in the ticket with a timestamp.",
    "If a parcel bound for {city} misses its scheduled scan by more than {hours} hours, the agent opens a "
    "trace request, notifies the recipient in plain language, and schedules a follow-up for the next shift.",
    "Refunds for {city} shipments above {amount} dollars need a supervisor code, a photo of the damage when "
    "one exists, and the original invoice number, and are paid to the original payment method within ten days.",
)
_SECTION_COUNT: Final = 32

SYSTEM_PROMPT: Final = "\n\n".join(
    (
        "You are the support assistant for Northwind Logistics. Follow the playbook below exactly.",
        *(
            " ".join(
                sentence.format(n=n, city=_CITIES[n % len(_CITIES)], hours=4 + (n % 5), amount=100 * (1 + (n % 7)))
                for sentence in _SECTION_SENTENCES
            )
            for n in range(1, _SECTION_COUNT + 1)
        ),
        f"When the user asks about the weather, call the {WEATHER_TOOL_NAME} tool with the city name. After a "
        "tool result arrives, answer in one short sentence that repeats the temperature exactly as the tool "
        "reported it.",
    )
)


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


def _chat_turn_one() -> Mapping[str, object]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": WEATHER_QUESTION},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": WEATHER_TOOL_NAME,
                    "description": _WEATHER_DESCRIPTION,
                    "parameters": dict(_WEATHER_PARAMETERS),
                },
            }
        ],
        "reasoning_effort": "low",
        "max_tokens": MAX_OUTPUT_TOKENS,
    }


def _chat_turn_two(turn_one: ParsedTurn) -> Mapping[str, object]:
    first: Final = _chat_turn_one()
    tool_call_id: Final = turn_one.tool_call.id if turn_one.tool_call is not None else ""
    return {
        **first,
        "messages": [
            *_message_list(first["messages"]),
            turn_one.assistant_echo,
            {"role": "tool", "tool_call_id": tool_call_id, "content": WEATHER_TOOL_RESULT},
        ],
    }


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


def _responses_turn_one() -> Mapping[str, object]:
    return {
        "instructions": SYSTEM_PROMPT,
        "input": [{"role": "user", "content": WEATHER_QUESTION}],
        "tools": [
            {
                "type": "function",
                "name": WEATHER_TOOL_NAME,
                "description": _WEATHER_DESCRIPTION,
                "parameters": dict(_WEATHER_PARAMETERS),
            }
        ],
        "reasoning": {"effort": "low"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def _responses_turn_two(turn_one: ParsedTurn) -> Mapping[str, object]:
    first: Final = _responses_turn_one()
    call_id: Final = turn_one.tool_call.id if turn_one.tool_call is not None else ""
    return {
        **first,
        "input": [
            *_message_list(first["input"]),
            *_message_list(turn_one.assistant_echo),
            {"type": "function_call_output", "call_id": call_id, "output": WEATHER_TOOL_RESULT},
        ],
    }


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


def _messages_turn_one() -> Mapping[str, object]:
    return {
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": WEATHER_QUESTION}],
        "tools": [
            {
                "name": WEATHER_TOOL_NAME,
                "description": _WEATHER_DESCRIPTION,
                "input_schema": dict(_WEATHER_PARAMETERS),
            }
        ],
        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        "max_tokens": MAX_OUTPUT_TOKENS,
    }


def _messages_turn_two(turn_one: ParsedTurn) -> Mapping[str, object]:
    first: Final = _messages_turn_one()
    tool_use_id: Final = turn_one.tool_call.id if turn_one.tool_call is not None else ""
    return {
        **first,
        "messages": [
            *_message_list(first["messages"]),
            {"role": "assistant", "content": turn_one.assistant_echo},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": WEATHER_TOOL_RESULT}],
            },
        ],
    }


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
    turn_one=_chat_turn_one,
    turn_two=_chat_turn_two,
    parse=_parse_chat,
)
RESPONSES_SCRIPT: Final = EndpointScript(
    endpoint="/v1/responses",
    route_type="aresponses",
    metadata_key="litellm_metadata",
    turn_one=_responses_turn_one,
    turn_two=_responses_turn_two,
    parse=_parse_responses,
)
MESSAGES_SCRIPT: Final = EndpointScript(
    endpoint="/v1/messages",
    route_type="anthropic_messages",
    metadata_key="litellm_metadata",
    turn_one=_messages_turn_one,
    turn_two=_messages_turn_two,
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


def _conversation_check(
    turn_one: TurnOutcome,
    parsed_one: ParsedTurn | str | None,
    turn_two: TurnOutcome | None,
    parsed_two: ParsedTurn | str | None,
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
    if turn_two is None or turn_two.error is not None:
        reason: Final = turn_two.error if turn_two is not None else "not sent"
        return CheckResult("conversation", "fail", f"turn 2 request with the tool result failed: {reason}")
    if isinstance(parsed_two, str):
        return CheckResult("conversation", "fail", f"turn 2 {parsed_two}")
    if parsed_two is None or "72" not in parsed_two.text:
        answer: Final = parsed_two.text[:160] if parsed_two is not None else ""
        return CheckResult(
            "conversation", "fail", f"turn 2 did not use the tool result {WEATHER_TOOL_RESULT!r}: {answer!r}"
        )
    return CheckResult(
        "conversation",
        "pass",
        f"turn 1 called {WEATHER_TOOL_NAME}({call.raw_arguments}), turn 2 answered {parsed_two.text[:120]!r}",
    )


def _cache_percent(parsed_two: ParsedTurn | None) -> float | None:
    if parsed_two is None or parsed_two.input_tokens <= 0:
        return None
    return round(100.0 * parsed_two.cache_read_tokens / parsed_two.input_tokens, 1)


def _missed_cache(parsed: ParsedTurn | str | None) -> bool:
    return isinstance(parsed, ParsedTurn) and parsed.cache_read_tokens == 0


def _hit_cache(parsed: ParsedTurn | str | None) -> bool:
    return isinstance(parsed, ParsedTurn) and parsed.cache_read_tokens > 0


def _prompt_caching_check(parsed_two: ParsedTurn | None, attempts: int) -> CheckResult:
    if parsed_two is None:
        return CheckResult("prompt_caching", "fail", "not checked, the conversation did not reach turn 2")
    percent: Final = _cache_percent(parsed_two)
    attempt_note: Final = " on the second attempt" if attempts > 1 else ""
    if parsed_two.cache_read_tokens > 0:
        return CheckResult(
            "prompt_caching",
            "pass",
            f"turn 2 read {parsed_two.cache_read_tokens} of {parsed_two.input_tokens} input tokens from cache "
            f"({percent}%){attempt_note}",
        )
    return CheckResult(
        "prompt_caching",
        "fail",
        f"turn 2 reported 0 cached tokens on {parsed_two.input_tokens} input tokens in {attempts} attempts, the "
        "provider did not reuse the repeated system prompt",
    )


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
            "cost_tracking", "warn", f"turn {turn_number} response_cost is $0, this model is priced at zero"
        )
    return None


def _cost_tracking_check(
    turn_one: TurnOutcome, turn_two: TurnOutcome | None, spend_rows: Mapping[str, float] | None
) -> CheckResult:
    turns: Final = tuple(turn for turn in (turn_one, turn_two) if turn is not None and turn.response is not None)
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
    ready_for_turn_two: Final = isinstance(parsed_one, ParsedTurn) and parsed_one.tool_call is not None
    turn_two: Final = (
        await _turn(script, script.turn_two(parsed_one), call, new_call_id())
        if ready_for_turn_two and isinstance(parsed_one, ParsedTurn)
        else None
    )
    first_parsed_two: Final = (
        script.parse(turn_two.response) if turn_two is not None and turn_two.response is not None else None
    )
    retried_turn_two: Final = (
        await _turn(script, script.turn_two(parsed_one), call, new_call_id())
        if isinstance(parsed_one, ParsedTurn) and _missed_cache(first_parsed_two)
        else None
    )
    retried_parsed_two: Final = (
        script.parse(retried_turn_two.response)
        if retried_turn_two is not None and retried_turn_two.response is not None
        else None
    )
    turn_two_used: Final = retried_turn_two if _hit_cache(retried_parsed_two) else turn_two
    parsed_two: Final = retried_parsed_two if _hit_cache(retried_parsed_two) else first_parsed_two
    cache_attempts: Final = 1 if retried_turn_two is None else 2
    answered: Final = tuple(
        turn for turn in (turn_one, turn_two_used) if turn is not None and turn.response is not None
    )
    spend_rows: Final = (
        await spend_lookup(tuple(turn.call_id for turn in answered)) if spend_lookup is not None and answered else None
    )
    second: Final = parsed_two if isinstance(parsed_two, ParsedTurn) else None
    checks: Final = (
        _conversation_check(turn_one, parsed_one, turn_two_used, parsed_two),
        _prompt_caching_check(second, cache_attempts),
        _cost_tracking_check(turn_one, turn_two_used, spend_rows),
    )
    return EndpointResult(
        endpoint=script.endpoint,
        status=_status(checks),
        checks=checks,
        cache_read_tokens=second.cache_read_tokens if second is not None else 0,
        cache_read_percent=_cache_percent(second),
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
