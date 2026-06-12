"""Raw-shape fidelity guards for the httpx-path shim family.

The default is the SAME composition as compat_sdk: explicit ``stream:
false`` first (the httpx path keeps the key on the wire — the xai/azure
precedent), then the openai guard with its full message-``name`` fallback
(nobody here strips names).

Two per-provider arms, reusing the compat_sdk implementations:

- heroku: ``_transform_messages`` flattens content lists
  (handle_messages_with_content_list_to_str_conversion — Heroku's API has
  no list-form content), LIVE on this path.
- minimax: ``remove_cache_control_flag_from_messages_and_tools`` is
  overridden to a no-op (markers reach the wire) while the v2 serializer
  strips them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from ...errors import TranslationError
from ..compat_sdk.guard import (
    cache_control_preserved,
    content_list_flattened,
    unsupported_request_shapes,
)
from .params import ALLOWED, CompatHttpxProvider

_Raw = Mapping[str, object]
_RawGuardFn = Callable[[_Raw], TranslationError | None]


def _heroku_guard(raw: _Raw) -> TranslationError | None:
    return content_list_flattened(raw, "heroku") or unsupported_request_shapes(raw)


def _minimax_guard(raw: _Raw) -> TranslationError | None:
    return cache_control_preserved(raw, "minimax") or unsupported_request_shapes(raw)


_OVERRIDES: Mapping[CompatHttpxProvider, _RawGuardFn] = MappingProxyType(
    {"heroku": _heroku_guard, "minimax": _minimax_guard}
)

GUARDS: Mapping[CompatHttpxProvider, _RawGuardFn] = MappingProxyType(
    {
        provider: _OVERRIDES.get(provider, unsupported_request_shapes)
        for provider in ALLOWED
    }
)
