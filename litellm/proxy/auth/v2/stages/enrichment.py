from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Dict, Optional

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth

# Loaders are injected so the mapping is unit-testable without a DB. The loaded
# row is only read by attribute, so ``object`` is enough and keeps this decoupled.
UserLoader = Callable[[str], Awaitable[Optional[object]]]
TeamLoader = Callable[[str], Awaitable[Optional[object]]]

# Source attr on the user/team row -> destination attr on the identity. The
# destination user_*/team_* slots are distinct from key-level fields, so this
# never overwrites a virtual key's own limits.
_USER_FIELD_MAP = {
    "max_budget": "user_max_budget",
    "tpm_limit": "user_tpm_limit",
    "rpm_limit": "user_rpm_limit",
    "spend": "user_spend",
}
_TEAM_FIELD_MAP = {
    "max_budget": "team_max_budget",
    "tpm_limit": "team_tpm_limit",
    "rpm_limit": "team_rpm_limit",
    "spend": "team_spend",
    "models": "team_models",
    "blocked": "team_blocked",
}


def _copy_missing(
    identity: UserAPIKeyAuth, source: object, field_map: Dict[str, str]
) -> None:
    for src_attr, dest_attr in field_map.items():
        value = getattr(source, src_attr, None)
        if value is not None and getattr(identity, dest_attr, None) is None:
            setattr(identity, dest_attr, value)


async def enrich_identity(
    identity: UserAPIKeyAuth,
    *,
    load_user: Optional[UserLoader] = None,
    load_team: Optional[TeamLoader] = None,
) -> UserAPIKeyAuth:
    """Populate the identity's user/team budget+limit fields from the user/team rows.

    Virtual keys arrive fully populated via ``get_key_object``; master/JWT/OAuth
    logins do not, so the existing pre-call budget/limit hooks read ``None`` and
    enforce nothing. This fills the gap, copying only fields that are unset (so it
    never overrides an already-resolved value) straight from the source rows.

    Mechanically faithful and additive; the exact enforcement still needs a live
    rate-limit check before it is trusted (see INTEGRATION.md).
    """
    user_id = getattr(identity, "user_id", None)
    if load_user is not None and user_id:
        user = await load_user(user_id)
        if user is not None:
            _copy_missing(identity, user, _USER_FIELD_MAP)

    team_id = getattr(identity, "team_id", None)
    if load_team is not None and team_id:
        team = await load_team(team_id)
        if team is not None:
            _copy_missing(identity, team, _TEAM_FIELD_MAP)

    return identity
