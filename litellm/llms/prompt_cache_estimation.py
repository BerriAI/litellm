from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import accumulate
from typing import Final, cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.integrations.anthropic_cache_control_hook import (
    CARRY_UNMATCHED_MESSAGE_POINTS,
    AnthropicCacheControlHook,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import parse_prompt_tokens_details
from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages
from litellm.llms.anthropic.common_utils import supports_anthropic_cache_control
from litellm.llms.anthropic.pass_through.messages.utils import prepare_native_messages
from litellm.llms.anthropic.prompt_cache_prediction import CountedBreakpoint, CountedPromptCachePlan
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.baseline_request import (
    BASELINE_PARAMETERS,
    CACHE_SETTINGS,
    NATIVE_ONLY_PARAMETERS,
    within_baseline_budget,
)
from litellm.types.llms.openai import AllMessageValues, ResponseInputParam
from litellm.types.utils import ModelInfo, PromptTokensDetailsWrapper, Usage
from litellm.utils import token_counter

_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_INPUT: Final[TypeAdapter[str | list[dict[str, JsonValue]]]] = TypeAdapter(str | list[dict[str, JsonValue]])
_SYSTEM: Final[TypeAdapter[str | list[dict[str, JsonValue]] | None]] = TypeAdapter(
    str | list[dict[str, JsonValue]] | None
)
_TTLS: Final = {"5m": 300, "30m": 1800, "1h": 3600, "24h": 86400}
_COUNT: Final = TypeAdapter(int | None)
_CONTROLS: Final = ("cache_control", "prompt_cache_breakpoint")


@dataclass(frozen=True, slots=True)
class EstimatedCachePlan:
    plan: CountedPromptCachePlan
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Part:
    value: JsonValue
    role: JsonValue
    end: bool
    control: Mapping[str, JsonValue]


def _parts(message: dict[str, JsonValue]) -> Iterator[_Part]:
    content: Final = message.get("content")
    role: Final = message.get("role")
    if not isinstance(content, list) or not content:
        yield _Part(message, role, True, message)
        return
    yield _Part({key: value for key, value in message.items() if key != "content"}, role, False, {})
    for index, part in enumerate(content):
        yield _Part(
            part,
            role,
            index == len(content) - 1,
            {
                key: message[key]
                if index == len(content) - 1 and message.get(key) is not None
                else part.get(key)
                if isinstance(part, dict)
                else None
                for key in _CONTROLS
            },
        )


def _tool_cache_control(tool: dict[str, JsonValue]) -> dict[str, JsonValue]:
    function: Final = tool.get("function")
    if tool.get("type") not in ("function", "custom") or "input_schema" in tool or not isinstance(function, dict):
        return tool
    control: Final = tool.get("cache_control")
    return {
        **tool,
        "function": {key: value for key, value in function.items() if key != "cache_control"},
        "cache_control": control if control is not None else function.get("cache_control"),
    }


def prepare_cache_request(
    kwargs: Mapping[str, object], model: str | None = None, provider: str | None = None, *, native: bool = False
) -> dict[str, JsonValue] | None:
    selected: Final = {
        key: kwargs[key] for key in (*BASELINE_PARAMETERS, "messages", "input", "extra_body") if key in kwargs
    }
    if not within_baseline_budget(selected):
        return None
    owned: Final = _OBJECT.validate_python(selected)
    if (
        model is None
        or provider is None
        or not (provider == "anthropic" or supports_anthropic_cache_control(model, provider))
    ):
        return owned
    try:
        user_agent: Final = AnthropicCacheControlHook.request_user_agent(kwargs)
        return _prepare_cache_injections(owned, model, provider, native=native, user_agent=user_agent)
    except (ValidationError, ValueError, TypeError):
        return None


def _prepare_cache_injections(
    request: dict[str, JsonValue], model: str, provider: str, *, native: bool, user_agent: str | None
) -> dict[str, JsonValue] | None:
    tools: Final = _MESSAGES.validate_python(request.get("tools") or [])
    transport: Final = (
        {"proxy_server_request": {"headers": {"user-agent": user_agent}}} if user_agent is not None else {}
    )
    if native:
        native_options: Final[dict[str, object]] = {**request, **transport}
        native_messages, system = prepare_native_messages(
            _MESSAGES.validate_python(request.get("messages")),
            _SYSTEM.validate_python(request.get("system")),
            native_options,
            model=model,
            custom_llm_provider=provider,
            tools=tools,
        )
        return (
            None
            if native_options.get("cache_control_injection_points")
            else _OBJECT.validate_python({**request, "messages": native_messages, "system": system})
        )
    hook: Final = AnthropicCacheControlHook()
    options: Final = {**(_responses_cache_input(request, hook, model) if "input" in request else request), **transport}
    messages: Final = cast(  # cast-ok: JSON validation retains provider extensions accepted by the shared hook
        list[AllMessageValues], _MESSAGES.validate_python(options.get("messages"))
    )
    AnthropicCacheControlHook.maybe_seed_default_injection_points(
        options,
        messages,
        model,
        provider,
        tools=tools,
        enable_prompt_caching=request.get("enable_prompt_caching") is True,
    )
    _, injected, remaining = hook.get_chat_completion_prompt(model, messages, options, None, None, {})
    if remaining.get("cache_control_injection_points"):
        return None
    return _OBJECT.validate_python(
        {
            **{key: value for key, value in request.items() if key not in ("input", "instructions")},
            "messages": injected,
        }
    )


def _responses_cache_input(
    request: dict[str, JsonValue], hook: AnthropicCacheControlHook, model: str
) -> dict[str, JsonValue]:
    original: Final = cast(  # cast-ok: the shared Responses adapter accepts JSON input extensions beyond SDK fields
        str | ResponseInputParam, _INPUT.validate_python(request["input"])
    )
    provisional: Final = ResponsesAPIRequestUtils.responses_input_to_chat_messages(original)
    _, marked, deferred = hook.get_chat_completion_prompt(
        model,
        provisional,
        {**request, CARRY_UNMATCHED_MESSAGE_POINTS: True},
        None,
        None,
        {},
    )
    merged: Final = ResponsesAPIRequestUtils.merge_prompt_management_input(original, provisional, marked)
    return _OBJECT.validate_python(
        {
            **_OBJECT.validate_python(deferred),
            "messages": resolve_structured_messages(None, {**request, "input": merged}),
        }
    )


def count_prefix_tokens(model: str, text: str) -> int:
    return token_counter(model=model, text=text)


def _weight(model: str, text: str, counter: Callable[[str, str], int]) -> int:
    return max(1, sum(counter(model, text[start : start + 8192]) for start in range(0, len(text), 8192)))


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ttl(control: object, default: int) -> int:
    value: Final = control.get("ttl") if isinstance(control, dict) else None
    return _TTLS.get(value, default) if isinstance(value, str) else default


def _cache_lifetime(request: Mapping[str, JsonValue], provider: str, prices: ModelInfo | None) -> int:
    retention: Final = request.get("prompt_cache_retention")
    requested: Final = _TTLS.get(retention) if provider in ("openai", "azure") and isinstance(retention, str) else None
    duration_pricing: Final = provider == "anthropic" or bool(
        prices and prices.get("cache_creation_input_token_cost_above_1hr") is not None
    )
    return _ttl(
        request.get("prompt_cache_options"),
        requested or (300 if duration_pricing else 600 if retention == "in_memory" else 1800),
    )


def _explicit_ttls(value: JsonValue) -> Iterator[int]:
    if isinstance(value, dict):
        yield _ttl(value.get("cache_control"), 0)
        for child in value.values():
            yield from _explicit_ttls(child)
    elif isinstance(value, list):
        for child in value:
            yield from _explicit_ttls(child)


def cache_ttl_upper_bound(request: Mapping[str, JsonValue] | None, provider: str, prices: ModelInfo | None) -> int:
    if request is None:
        return 3600 if provider == "anthropic" else max(_TTLS.values())
    return max((_cache_lifetime(request, provider, prices), *_explicit_ttls(dict(request))))


def estimate_cache_plan(
    request: dict[str, JsonValue],
    model: str,
    provider: str,
    prices: ModelInfo | None,
    usage: Usage,
    counter: Callable[[str, str], int] = count_prefix_tokens,
) -> EstimatedCachePlan | None:
    if any(
        request.get(key)
        for key in (
            "previous_response_id",
            "conversation",
            "cached_content",
            *NATIVE_ONLY_PARAMETERS,
        )
    ):
        return None
    supplied: Final = request.get("messages") or request.get("input")
    try:
        input_messages: Final = _MESSAGES.validate_python(
            supplied if isinstance(supplied, list) else resolve_structured_messages(None, {"input": supplied})
        )
        tools: Final = [_tool_cache_control(tool) for tool in _MESSAGES.validate_python(request.get("tools") or [])]
    except (ValidationError, TypeError):
        return None
    prefix: Final = tuple(
        {"role": "system", "content": content}
        for content in (tools, request.get("system"), request.get("instructions"))
        if content
    )
    messages: Final = _MESSAGES.validate_python((*prefix, *input_messages))
    if (
        not messages
        or len(messages)
        + sum(len(content) for content in (item.get("content") for item in messages) if isinstance(content, list))
        > 2048
    ):
        return None
    parts: Final = tuple(
        part for message in messages for part in _parts(message)
    )  # comprehension-ok: flatten bounded blocks
    settings: Final = _json(
        {
            key: request[key]
            for key in CACHE_SETTINGS
            if key in request and key not in ("tools", "system", "instructions")
        }
    )
    serialized: Final = tuple(
        _json(
            {key: value for key, value in part.value.items() if key not in _CONTROLS}
            if isinstance(part.value, dict)
            else part.value
        )
        for part in parts
    )
    weights: Final = tuple(
        accumulate((_weight(model, settings, counter), *(_weight(model, part, counter) for part in serialized)))
    )
    hashes: Final = tuple(
        accumulate(
            serialized,
            lambda prior, item: hashlib.sha256((prior + item).encode()).hexdigest(),
            initial=hashlib.sha256(settings.encode()).hexdigest(),
        )
    )[1:]
    options: Final = _OBJECT.validate_python(request.get("prompt_cache_options") or {})
    retention: Final = request.get("prompt_cache_retention")
    requested_ttl: Final = (
        _TTLS.get(retention) if provider in ("openai", "azure") and isinstance(retention, str) else None
    )
    duration_pricing: Final = provider == "anthropic" or bool(
        prices and prices.get("cache_creation_input_token_cost_above_1hr") is not None
    )
    lifetime: Final = _cache_lifetime(request, provider, prices)
    implicit: Final = not duration_pricing and options.get("mode") != "explicit"
    initial: Final = next(
        (index - 1 for index, part in enumerate(parts) if part.role not in ("system", "developer")), len(parts) - 1
    )
    user_boundaries: Final = provider in ("openai", "azure") and bool(
        prices and prices.get("supports_prompt_cache_breakpoint")
    )
    eligible: Final = (
        tuple(
            index
            for index, part in enumerate(parts)
            if part.end
            and (
                not user_boundaries
                or part.role == "user"
                or index == initial
                or part.role == "tool"
                and (index + 1 == len(parts) or parts[index + 1].role != "tool")
            )
        )
        if implicit
        else ()
    )
    automatic: Final = (len(parts) - 1,) if isinstance(request.get("cache_control"), dict) else ()
    explicit: Final = (
        *tuple(index for index, part in enumerate(parts) if any(part.control.get(key) for key in _CONTROLS)),
        *automatic,
    )
    enabled: Final = prices is not None and (
        prices.get("cache_read_input_token_cost") is not None or prices.get("supports_prompt_caching") is True
    )
    boundaries: Final = tuple(sorted({*explicit, *eligible[-1:]}))[-4:] if enabled else ()

    def marker(index: int) -> CountedBreakpoint:
        candidates: Final = tuple(sorted({index, *explicit[:2], *explicit[-50:], initial, *eligible[-21:]}))
        lookback: Final = tuple(hashes[earlier] for earlier in candidates if 0 <= earlier <= index)
        return CountedBreakpoint(
            fingerprint=hashes[index],
            content_fingerprint=hashes[index],
            ttl_seconds=_ttl(
                parts[index].control.get("cache_control")
                or (request.get("cache_control") if index in automatic else None),
                lifetime,
            ),
            prefix_tokens=min(usage.prompt_tokens, round(usage.prompt_tokens * weights[index + 1] / weights[-1])),
            lookback_fingerprints=lookback,
            lookback_content_fingerprints=lookback,
        )

    return EstimatedCachePlan(
        CountedPromptCachePlan(usage.prompt_tokens, tuple(marker(index) for index in boundaries)),
        (
            "cold_cache_at_session_start",
            "reported_input_tokens_scaled_across_prefixes",
            "same_output_tokens",
            "message_boundary_cache_approximation",
            "cache_tokens_scaled_across_modalities",
            *(
                ()
                if duration_pricing
                or requested_ttl
                or isinstance(options.get("ttl"), str)
                and options.get("ttl") in _TTLS
                else ("cache_lifetime_assumed_10m" if retention == "in_memory" else "cache_lifetime_assumed_30m",)
            ),
        ),
    )


def prepare_baseline_usage(usage: Usage | None, provider: str, request: Mapping[str, JsonValue] | None) -> Usage | None:
    if usage is None or request is None or provider != "anthropic":
        return usage
    return usage.model_copy(update={key: request.get(key) for key in ("speed", "inference_geo")})


def normalize_cache_usage(usage: Usage) -> Usage:
    details: Final = usage.prompt_tokens_details or PromptTokensDetailsWrapper()
    read: Final = details.cached_tokens or 0
    write: Final = (
        _COUNT.validate_python(getattr(details, "cache_creation_tokens", None))
        or _COUNT.validate_python(getattr(details, "cache_write_tokens", None))
        or 0
    )
    parsed: Final = parse_prompt_tokens_details(usage)
    other: Final = parsed["audio_tokens"] + parsed["image_tokens"] + parsed["video_tokens"]
    cached: Final = getattr(details, "cached_tokens_details", None)
    cached_text: Final = (cached.text_tokens or 0) if cached is not None else 0
    return usage.model_copy(
        update={
            "prompt_tokens_details": details.model_copy(
                update={
                    "text_tokens": max(usage.prompt_tokens - read - write - other, 0) + cached_text,
                    "cached_tokens": read,
                    "cache_creation_tokens": write,
                }
            )
        }
    )
