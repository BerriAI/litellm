"""Raw-shape fidelity guards for the SDK-path openai-compat family.

The shared explicit ``stream: false`` arm runs first: on this path
``completion()`` forwards the caller's False into ``get_optional_params``
(non-default against the ``None`` default), it lands in optional_params,
and the SDK serializes the key onto the wire — and the compat_httpx
family (whose guard.py composes this default; cometapi moved there at the
sibling merge) keeps the key on the wire the same way, so the arm
covers both families — while the IR cannot represent absent-vs-false
(verified in-process at HEAD for both; the same arm the azure and xai
guards compose).

The openai guard then runs with its full message-``name`` fallback: none of
the family configs strips names (only xai does), so v1 forwards ``name``
verbatim on every role.

Two per-provider arms (the ``GUARDS`` table carries them; everyone else gets
the shared default):

- ``cache_control`` preserved: dashscope and zai OVERRIDE
  ``remove_cache_control_flag_from_messages_and_tools`` to keep the markers
  on the wire, while the v2 openai_compat serializer strips them like the
  base transform — so any marker-bearing request falls back (the azure guard
  precedent).
- content-list flatten: docker_model_runner and publicai run
  ``handle_messages_with_content_list_to_str_conversion`` inside
  ``_transform_messages`` (LIVE on the SDK path), so ANY list-form message
  content falls back; the shared guard's single-text-list arm only covers
  the one-block case.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import cast

from ...errors import TranslationError
from ..openai_compat.guard import carries_cache_control, explicit_stream_false
from ..openai_compat.guard import (
    unsupported_request_shapes as openai_unsupported_request_shapes,
)
from .params import ALLOWED, CompatSdkProvider

_Raw = Mapping[str, object]
_RawGuardFn = Callable[[_Raw], TranslationError | None]


def unsupported_request_shapes(raw: _Raw) -> TranslationError | None:
    return explicit_stream_false(raw) or openai_unsupported_request_shapes(raw)


def cache_control_preserved(raw: _Raw, provider: str) -> TranslationError | None:
    for field in ("messages", "tools"):
        if carries_cache_control(raw.get(field)):
            return TranslationError.of_unsupported(
                f"cache_control inside {field} ({provider} preserves it on the "
                "wire — remove_cache_control_flag_from_messages_and_tools is "
                "overridden to a no-op — while the v2 serializer strips it); "
                "v1 serves it"
            )
    return None


def content_list_flattened(raw: _Raw, provider: str) -> TranslationError | None:
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return None
    for message in cast(Sequence[object], messages):
        if isinstance(message, Mapping) and isinstance(
            cast(Mapping[str, object], message).get("content"), list
        ):
            return TranslationError.of_unsupported(
                f"list-form message content ({provider} flattens content "
                "lists to strings inside _transform_messages, live on the "
                "SDK path); v1 serves its flatten"
            )
    return None


def _with_cache_control_arm(provider: str) -> _RawGuardFn:
    def guard(raw: _Raw) -> TranslationError | None:
        return cache_control_preserved(raw, provider) or unsupported_request_shapes(raw)

    return guard


def _with_content_list_arm(provider: str) -> _RawGuardFn:
    def guard(raw: _Raw) -> TranslationError | None:
        return content_list_flattened(raw, provider) or unsupported_request_shapes(raw)

    return guard


_OVERRIDES: Mapping[CompatSdkProvider, _RawGuardFn] = MappingProxyType(
    {
        "dashscope": _with_cache_control_arm("dashscope"),
        "zai": _with_cache_control_arm("zai"),
        "docker_model_runner": _with_content_list_arm("docker_model_runner"),
        "publicai": _with_content_list_arm("publicai"),
    }
)

GUARDS: Mapping[CompatSdkProvider, _RawGuardFn] = MappingProxyType(
    {
        provider: _OVERRIDES.get(provider, unsupported_request_shapes)
        for provider in ALLOWED
    }
)
"""The COMPLETE family guard table (one row per registered provider);
``engine/pipeline.py`` splices it whole, one line for the family."""
