"""Serve `/model/info` and `/model_group/info` from memory for a short while.

Both routes rebuild their answer on every call, on the worker's event loop: a deep
copy of every deployment, the team table parsed row by row, pricing lookups per
model. With `general_settings.model_listing_cache_ttl_seconds` set, a worker
answers the same caller and query from memory until the TTL passes or the
router's registry version moves, which it does on every deployment change.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Final, Generic, TypeVar

from pydantic import TypeAdapter

from litellm.proxy._types import UserAPIKeyAuth

MODEL_LISTING_CACHE_TTL_SETTING: Final = "model_listing_cache_ttl_seconds"
_MAX_ENTRIES: Final = 256
_KEY_MODELS: Final = TypeAdapter(tuple[str, ...])

_Listing: Final = TypeVar("_Listing")


def listing_cache_ttl(general_settings: Mapping[str, object]) -> float:
    """Seconds a listing is served from memory; 0 when the setting is absent, zero or malformed."""
    raw: Final = general_settings.get(MODEL_LISTING_CACHE_TTL_SETTING)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return 0.0
    return float(raw)


def listing_cache_key(route: str, user_api_key_dict: UserAPIKeyAuth, params: Mapping[str, object]) -> str:
    """One entry per caller identity and query; a listing is filtered by who asks and how."""
    scope: Final = (
        user_api_key_dict.token,
        user_api_key_dict.team_id,
        user_api_key_dict.user_id,
        user_api_key_dict.org_id,
        str(user_api_key_dict.user_role),
        _KEY_MODELS.validate_python(user_api_key_dict.models or ()),  # pyright: ignore[reportUnknownMemberType]  # key.models is an untyped String[] column
    )
    return repr((route, scope, tuple(sorted(params.items()))))


class ModelListingCache(Generic[_Listing]):
    """Listings of one route, each kept until its TTL passes or the registry version it was built from moves."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock: Final = clock
        self._entries: Final[dict[str, tuple[float, int, _Listing]]] = {}  # mutable-ok: the cache is the state

    def get(self, key: str, registry_version: int) -> _Listing | None:
        entry: Final = self._entries.get(key)
        if entry is None:
            return None
        expires_at, version, listing = entry
        if version != registry_version or self._clock() > expires_at:
            self._entries.pop(key, None)
            return None
        return listing

    def put(self, key: str, listing: _Listing, registry_version: int, ttl_seconds: float) -> None:
        if len(self._entries) >= _MAX_ENTRIES:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = (self._clock() + ttl_seconds, registry_version, listing)

    def clear(self) -> None:
        """Forget every listing: the router was replaced, so its registry versions start over."""
        self._entries.clear()
