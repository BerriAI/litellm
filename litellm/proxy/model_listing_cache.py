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
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

from pydantic import TypeAdapter

from litellm.proxy._types import UserAPIKeyAuth

MODEL_LISTING_CACHE_TTL_SETTING: Final = "model_listing_cache_ttl_seconds"
_MAX_ENTRIES: Final = 256
_KEY_NAMES: Final = TypeAdapter(tuple[str, ...])
_KEY_ALIASES: Final = TypeAdapter(dict[str, object])

_Listing: Final = TypeVar("_Listing")


@dataclass(frozen=True)
class ListingScope:
    """One cached listing: the caller and query it answers, and how long it is served."""

    key: str
    ttl_seconds: float


def listing_cache_ttl(general_settings: Mapping[str, object]) -> float:
    """Seconds a listing is served from memory; 0 when the setting is absent, zero or malformed."""
    raw: Final = general_settings.get(MODEL_LISTING_CACHE_TTL_SETTING)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return 0.0
    return float(raw)


def listing_cache_key(route: str, user_api_key_dict: UserAPIKeyAuth, params: Mapping[str, object]) -> str:
    """One entry per caller and query.

    The scope holds every field the listing is filtered by: the credential hash, the
    team, user, organization, role and agent, the user and team that invoked the agent, the
    key's and the team's models and access groups, and the aliases the names are read through.
    """
    scope: Final = (
        user_api_key_dict.token,
        user_api_key_dict.team_id,
        user_api_key_dict.user_id,
        user_api_key_dict.org_id,
        user_api_key_dict.agent_id,
        user_api_key_dict.agent_caller,
        str(user_api_key_dict.user_role),
        _names(user_api_key_dict.models),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # key.models is an untyped String[] column
        _names(user_api_key_dict.team_models),
        _names(user_api_key_dict.access_group_ids),
        _aliases(user_api_key_dict.aliases),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # key.aliases is an untyped Json column
        _aliases(user_api_key_dict.team_model_aliases),
    )
    return repr((route, scope, tuple(sorted(params.items()))))


def listing_cache_scope(
    general_settings: Mapping[str, object], route: str, user_api_key_dict: UserAPIKeyAuth, params: Mapping[str, object]
) -> ListingScope | None:
    """None when the setting is off, or when the caller has no credential hash.

    A JWT subject or a custom-auth caller is told apart only by its claims, so it is
    computed every call rather than risk serving it another subject's listing.
    """
    ttl_seconds: Final = listing_cache_ttl(general_settings)
    if not ttl_seconds or user_api_key_dict.token is None:
        return None
    return ListingScope(listing_cache_key(route, user_api_key_dict, params), ttl_seconds)


def _names(column: object) -> tuple[str, ...]:
    return _KEY_NAMES.validate_python(column or ())


def _aliases(column: object) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(_KEY_ALIASES.validate_python(column or {}).items(), key=lambda alias: alias[0]))


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
