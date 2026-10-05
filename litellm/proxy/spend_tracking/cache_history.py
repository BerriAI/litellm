from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages
from litellm.llms.anthropic.prompt_cache_prediction import CountedBreakpoint, CountedPromptCachePlan
from litellm.llms.prompt_cache_policy import CacheHistoryPolicy, cache_history_policy, cache_ttl
from litellm.types.utils import CacheCreationTokenDetails, ModelInfo, PromptTokensDetailsWrapper, Usage
from litellm.utils import token_counter

_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_SETTINGS: Final = (
    "system",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "text",
    "reasoning",
    "reasoning_effort",
    "thinking",
    "verbosity",
    "prompt_cache_key",
    "cache_key",
    "cached_content",
    "previous_response_id",
    "conversation",
    "context_management",
    "compaction",
)
_REQUEST_KEYS: Final = (*_SETTINGS, "messages", "input", "prompt_cache_options", "prompt_cache_retention")
_MAX_BYTES: Final = 4 * 1024 * 1024
_MAX_NODES: Final = 32768
_MAX_PARTS: Final = 2048
_MAX_DEPTH: Final = 32
_TOKEN_CHUNK: Final = 8192


def count_prefix_tokens(model: str, text: str) -> int:
    return token_counter(model=model, text=text)


@dataclass(frozen=True, slots=True)
class _Part:
    value: JsonValue = field(repr=False)
    role: JsonValue
    end: bool
    control: JsonValue = None


@dataclass(frozen=True, slots=True)
class PreparedCacheRequest:
    parts: tuple[_Part, ...] = field(repr=False)
    settings: Mapping[str, JsonValue] = field(repr=False)
    policy: CacheHistoryPolicy
    enabled: bool

    def count(self, model: str, counter: Callable[[str, str], int] = count_prefix_tokens) -> EstimatedCacheRequest:
        serialized: Final = tuple(_json(_without_controls(part.value)) for part in self.parts)
        settings: Final = _json(self.settings)
        weights: Final = tuple(
            accumulate((_weight(model, settings, counter), *(_weight(model, part, counter) for part in serialized)))
        )
        digests: Final = tuple(
            accumulate(
                serialized,
                lambda prior, item: hashlib.sha256((prior + item).encode()).hexdigest(),
                initial=hashlib.sha256(settings.encode()).hexdigest(),
            )
        )
        initial_end: Final = next(
            (index - 1 for index, part in enumerate(self.parts) if part.role not in ("developer", "system")),
            len(self.parts) - 1,
        )
        prefixes: Final = tuple(
            _Prefix(
                fingerprint=digests[index + 1],
                weight=weights[index + 1],
                explicit=isinstance(part.control, dict)
                and bool(part.control.get("prompt_cache_breakpoint") or part.control.get("cache_control")),
                eligible=part.end
                and self.policy.eligible(
                    part.role,
                    self.parts[index + 1].role if index + 1 < len(self.parts) else None,
                    index == initial_end,
                ),
                ttl=cache_ttl(part.control.get("cache_control"), self.policy.lifetime)
                if isinstance(part.control, dict)
                else self.policy.lifetime,
                initial=index == initial_end,
            )
            for index, part in enumerate(self.parts)
        )
        return EstimatedCacheRequest(
            prefixes=prefixes,
            weight=weights[-1],
            implicit=self.policy.implicit,
            enabled=self.enabled,
            assumptions=(
                "cold_cache_at_session_start",
                "reported_input_tokens_scaled_across_prefixes",
                "same_output_tokens",
                "supplied_request_settings_with_baseline_overrides",
                *self.policy.assumptions,
                "message_boundary_cache_approximation",
                "chunked_prefix_token_weights",
            ),
        )


def _weight(model: str, text: str, counter: Callable[[str, str], int]) -> int:
    return max(
        1,
        sum(counter(model, text[start : start + _TOKEN_CHUNK]) for start in range(0, len(text), _TOKEN_CHUNK)),
    )


def _json_cost(value: object, depth: int = 0) -> Iterator[int]:
    if depth > _MAX_DEPTH:
        yield _MAX_BYTES + 1
    elif isinstance(value, str):
        yield (6 if value.isascii() else 12) * len(value) + 2
    elif isinstance(value, dict):
        yield 2
        for key, item in cast(dict[object, object], value).items():
            yield from _json_cost(key, depth + 1)
            yield from _json_cost(item, depth + 1)
            yield 2
    elif isinstance(value, (list, tuple)):
        yield 2
        for item in cast(list[object] | tuple[object, ...], value):
            yield from _json_cost(item, depth + 1)
            yield 1
    elif isinstance(value, int) and value.bit_length() > 64:
        yield _MAX_BYTES + 1
    elif value is None or isinstance(value, (bool, int, float)):
        yield 32
    else:
        yield _MAX_BYTES + 1


def _within_budget(value: object) -> bool:
    return all(
        size <= _MAX_BYTES and nodes <= _MAX_NODES for nodes, size in enumerate(accumulate(_json_cost(value)), 1)
    )


def _selected(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    mapping: Final = cast(Mapping[str, object], value)
    return {key: mapping[key] for key in _REQUEST_KEYS if key in mapping}


@dataclass(frozen=True, slots=True)
class _Prefix:
    fingerprint: str
    weight: int
    explicit: bool
    eligible: bool
    ttl: int
    initial: bool


@dataclass(frozen=True, slots=True)
class EstimatedCacheRequest:
    prefixes: tuple[_Prefix, ...]
    weight: int
    implicit: bool
    enabled: bool
    assumptions: tuple[str, ...]

    def plan(self, usage: Usage) -> CountedPromptCachePlan:
        eligible: Final = tuple(index for index, prefix in enumerate(self.prefixes) if prefix.eligible)
        implicit: Final = eligible[-1:] if self.implicit else ()
        explicit: Final = tuple(index for index, prefix in enumerate(self.prefixes) if prefix.explicit)
        boundaries: Final = tuple(sorted(set((*explicit, *implicit))))[-4:] if self.enabled else ()
        return CountedPromptCachePlan(
            usage.prompt_tokens,
            tuple(self._marker(index, usage.prompt_tokens) for index in boundaries),
        )

    def _marker(self, index: int, total: int) -> CountedBreakpoint:
        prefix: Final = self.prefixes[index]
        candidates: Final = self.prefixes[: index + 1]
        explicit: Final = tuple(item for item in candidates if item.explicit)
        implicit: Final = tuple(item for item in candidates if item.eligible) if self.implicit else ()
        lookback: Final = tuple(
            item.fingerprint
            for item in (
                *explicit[:2],
                *explicit[-50:],
                *(item for item in implicit if item.initial),
                *implicit[-21:],
                prefix,
            )
        )
        return CountedBreakpoint(
            fingerprint=prefix.fingerprint,
            content_fingerprint=prefix.fingerprint,
            ttl_seconds=prefix.ttl,
            prefix_tokens=min(total, round(total * prefix.weight / max(1, self.weight))),
            lookback_fingerprints=lookback,
            lookback_content_fingerprints=lookback,
        )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parts(message: dict[str, JsonValue]) -> Iterator[_Part]:
    content: Final = message.get("content")
    role: Final = message.get("role")
    if not isinstance(content, list) or not content:
        yield _Part(message, role, True, message)
        return
    yield _Part({key: value for key, value in message.items() if key != "content"}, role, False)
    for index, part in enumerate(content):
        yield _Part(part, role, index == len(content) - 1, part)


def _without_controls(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _without_controls(item)
            for key, item in value.items()
            if key not in ("cache_control", "prompt_cache_breakpoint")
        }
    if isinstance(value, list):
        return [_without_controls(item) for item in value]
    return value


def prepare_cache_request(
    kwargs: Mapping[str, object],
    model: str,
    provider: str,
    prices: ModelInfo | None,
    baseline_params: Mapping[str, object],
) -> PreparedCacheRequest | None:
    request: Final = _selected(kwargs)
    extra: Final = _selected(kwargs.get("extra_body"))
    baseline: Final = _selected(baseline_params)
    baseline_extra: Final = _selected(baseline_params.get("extra_body"))
    if not _within_budget((request, extra, baseline, baseline_extra)):
        return None
    supplied: Final = request.get("messages")
    messages: Final = resolve_structured_messages(_MESSAGES.validate_python(supplied) if supplied else None, request)
    if not messages or not _within_budget(messages):
        return None
    combined: Final = {
        **request,
        **extra,
        **{key: value for key, value in baseline.items() if value is not None},
        **baseline_extra,
    }
    settings: Final = _OBJECT.validate_python({key: combined[key] for key in _SETTINGS if key in combined})
    if not _within_budget((messages, settings)):
        return None
    if any(settings.get(key) for key in ("previous_response_id", "conversation", "cached_content")):
        return None
    options: Final = _OBJECT.validate_python(combined.get("prompt_cache_options") or {})
    policy: Final = cache_history_policy(model, provider, prices, options, combined.get("prompt_cache_retention"))
    validated: Final = _MESSAGES.validate_python(messages)
    part_count: Final = sum(
        len(content) + 1 if isinstance(content, list) else 1
        for content in (message.get("content") for message in validated)
    )
    if part_count > _MAX_PARTS:
        return None
    parts: Final = tuple(
        part for message in validated for part in _parts(message)
    )  # comprehension-ok: flatten bounded message blocks
    return PreparedCacheRequest(
        parts=parts,
        settings=settings,
        policy=policy,
        enabled=prices is not None
        and (prices.get("cache_read_input_token_cost") is not None or prices.get("supports_prompt_caching") is True),
    )


def capture_cache_request(
    kwargs: Mapping[str, object],
    model: str,
    provider: str,
    prices: ModelInfo | None,
    baseline_params: Mapping[str, object],
) -> EstimatedCacheRequest | None:
    prepared: Final = prepare_cache_request(kwargs, model, provider, prices, baseline_params)
    return prepared.count(model) if prepared else None


class _CacheDetails(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_write_tokens: int | None = None
    cache_creation_token_details: CacheCreationTokenDetails | None = None


def normalize_cache_usage(usage: Usage) -> Usage:
    details: Final = usage.prompt_tokens_details or PromptTokensDetailsWrapper()
    parsed: Final = _CacheDetails.model_validate(details)
    other_input: Final = (details.audio_tokens or 0) + (details.image_tokens or 0) + (details.video_tokens or 0)
    read: Final = parsed.cached_tokens or 0
    write: Final = parsed.cache_creation_tokens or parsed.cache_write_tokens or 0
    return usage.model_copy(
        update={
            "prompt_tokens_details": details.model_copy(
                update={
                    "text_tokens": usage.prompt_tokens - read - write - other_input,
                    "cached_tokens": read,
                    "cache_creation_tokens": write,
                    "cache_creation_token_details": parsed.cache_creation_token_details,
                }
            )
        }
    )
