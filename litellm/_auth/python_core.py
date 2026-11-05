from datetime import datetime
from typing import Final

from litellm._auth.decision import (
    AdminKey,
    Allow,
    CachedAdminKey,
    Deny,
    EvictPresentedKey,
    ExpiredKey,
    LoadKey,
    MalformedKey,
    MasterKey,
    MasterKeyOnlyRoute,
    MissingKey,
    NoDatabase,
    NoMasterKey,
    NonAdminKey,
    NotAVirtualKey,
    PresentedKey,
    RequestFacts,
    ResolveEndUser,
    ResolvePresentedKey,
    Settings,
    Step,
    UserAuthDisabled,
    UserAuthRoute,
    VirtualKey,
)
from litellm._auth.key import hash_token, is_master_key

_USER_AUTH_ROUTE: Final = "/user/auth"
_VIRTUAL_KEY_PREFIX: Final = "sk-"


class PythonCore:
    def start(self, facts: RequestFacts, settings: Settings, now: datetime) -> Step:
        token: Final = facts.token
        if settings.master_key is None:
            return Allow(NoMasterKey(token=token))
        if token is None:
            return Deny(MissingKey())
        if token == "":
            return Deny(MalformedKey())
        if facts.route == _USER_AUTH_ROUTE:
            return Allow(UserAuthRoute()) if settings.allow_user_auth else Deny(UserAuthDisabled())
        if is_master_key(token, settings.master_key):
            return ResolveEndUser(prefetch_key=False, resume=lambda: Allow(MasterKey()))
        hashed_token: Final = hash_token(token)
        return ResolveEndUser(
            prefetch_key=True,
            resume=lambda: ResolvePresentedKey(
                hashed_token=hashed_token,
                resume=lambda presented: _after_presented_key(presented, token, hashed_token, facts, settings, now),
            ),
        )


def _after_presented_key(
    presented: PresentedKey | None,
    token: str,
    hashed_token: str,
    facts: RequestFacts,
    settings: Settings,
    now: datetime,
) -> Step:
    match presented:
        case AdminKey(expires=expires) if expires is not None and expires < now:
            return EvictPresentedKey(
                hashed_token=hashed_token, resume=lambda: Deny(ExpiredKey(expires=expires, now=now))
            )
        case AdminKey():
            return Allow(CachedAdminKey())
        case NonAdminKey() | None:
            pass
    if facts.route in settings.master_key_only_routes:
        return Deny(MasterKeyOnlyRoute())
    if not settings.has_db:
        return Deny(NoDatabase())
    if presented is not None:
        return Allow(VirtualKey(loaded_from_db=False))
    if not token.startswith(_VIRTUAL_KEY_PREFIX):
        return Deny(NotAVirtualKey())
    return LoadKey(hashed_token=hashed_token, resume=lambda: Allow(VirtualKey(loaded_from_db=True)))
