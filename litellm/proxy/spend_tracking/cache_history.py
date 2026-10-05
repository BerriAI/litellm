from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages
from litellm.llms.anthropic.prompt_cache_prediction import CountedBreakpoint, CountedPromptCachePlan
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
_TTLS: Final = MappingProxyType({"5m": 300, "30m": 1800, "1h": 3600, "24h": 86400})


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


def _ttl(control: JsonValue, default: int) -> int:
    if not isinstance(control, dict):
        return default
    value: Final = control.get("ttl")
    return _TTLS.get(value, default) if isinstance(value, str) else default


def _parts(message: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    content: Final = message.get("content")
    if not isinstance(content, list) or not content:
        return (dict(message),)
    envelope: Final = {key: value for key, value in message.items() if key != "content"}
    return tuple({**envelope, "content": part} for part in content)


def _control(part: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    content: Final = part.get("content")
    return content if isinstance(content, dict) else part


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


def capture_cache_request(
    kwargs: Mapping[str, object],
    model: str,
    provider: str,
    prices: ModelInfo | None,
    baseline_params: Mapping[str, object],
) -> EstimatedCacheRequest | None:
    supplied: Final = kwargs.get("messages")
    messages: Final = resolve_structured_messages(
        _MESSAGES.validate_python(supplied) if supplied else None, dict(kwargs)
    )
    if not messages:
        return None
    extra: Final = _OBJECT.validate_python(kwargs.get("extra_body") or {})
    baseline_extra: Final = _OBJECT.validate_python(baseline_params.get("extra_body") or {})
    combined: Final = {
        **kwargs,
        **extra,
        **{key: value for key, value in baseline_params.items() if value is not None},
        **baseline_extra,
    }
    settings: Final = _OBJECT.validate_python({key: combined[key] for key in _SETTINGS if key in combined})
    if any(settings.get(key) for key in ("previous_response_id", "conversation", "cached_content")):
        return None
    options: Final = _OBJECT.validate_python(combined.get("prompt_cache_options") or {})
    anthropic: Final = provider == "anthropic" or "claude" in model
    modern_openai: Final = (
        provider in ("openai", "azure")
        and prices is not None
        and prices.get("supports_prompt_cache_breakpoint") is True
    )
    retention: Final = combined.get("prompt_cache_retention")
    default_lifetime: Final = 300 if anthropic else 600 if retention == "in_memory" else 1800
    lifetime: Final = _ttl(options, default_lifetime)
    lifetime_known: Final = (
        (isinstance(options.get("ttl"), str) and options.get("ttl") in _TTLS) or anthropic or modern_openai
    )

    implicit: Final = not anthropic and options.get("mode") != "explicit"
    groups: Final = tuple(_parts(message) for message in _MESSAGES.validate_python(messages))
    parts: Final = tuple(
        part for group in groups for part in group
    )  # comprehension-ok: flatten normalized message blocks
    ends: Final = frozenset(index - 1 for index in accumulate(len(group) for group in groups))
    initial_end: Final = next(
        (index - 1 for index, part in enumerate(parts) if part.get("role") not in ("developer", "system")),
        len(parts) - 1,
    )
    serialized: Final = tuple(_json(_without_controls(part)) for part in parts)
    weights: Final = tuple(
        accumulate(
            (
                max(1, token_counter(model=model, text=_json(settings))),
                *(max(1, token_counter(model=model, text=part)) for part in serialized),
            )
        )
    )
    digests: Final = tuple(
        accumulate(
            serialized,
            lambda prior, item: hashlib.sha256((prior + item).encode()).hexdigest(),
            initial=hashlib.sha256(_json(settings).encode()).hexdigest(),
        )
    )
    prefixes: Final = tuple(
        _Prefix(
            fingerprint=digests[index + 1],
            weight=weights[index + 1],
            explicit=bool(_control(part).get("prompt_cache_breakpoint") or _control(part).get("cache_control")),
            eligible=index in ends
            and (
                not modern_openai
                or part.get("role") == "user"
                or (part.get("role") == "tool" and (index == len(parts) - 1 or parts[index + 1].get("role") != "tool"))
                or index == initial_end
            ),
            ttl=_ttl(_control(part).get("cache_control"), lifetime),
            initial=index == initial_end,
        )
        for index, part in enumerate(parts)
    )
    return EstimatedCacheRequest(
        prefixes=prefixes,
        weight=weights[-1],
        implicit=implicit,
        enabled=prices is not None
        and (prices.get("cache_read_input_token_cost") is not None or prices.get("supports_prompt_caching") is True),
        assumptions=(
            "cold_cache_at_session_start",
            "reported_input_tokens_scaled_across_prefixes",
            "same_output_tokens",
            "supplied_request_settings_with_baseline_overrides",
            *(
                ("cache_lifetime_assumed_10m",)
                if retention == "in_memory"
                else ("cache_lifetime_assumed_30m",)
                if not lifetime_known
                else ()
            ),
            "message_boundary_cache_approximation",
        ),
    )


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
