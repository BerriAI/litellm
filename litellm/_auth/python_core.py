from typing import Final

from litellm._auth.decision import (
    Allow,
    Defer,
    Deny,
    DenyReason,
    MasterKey,
    NoMasterKey,
    Outcome,
    RequestFacts,
    Settings,
    UserAuthRoute,
)
from litellm._auth.key import is_master_key

_USER_AUTH_ROUTE: Final = "/user/auth"


class PythonCore:
    def decide(self, facts: RequestFacts, settings: Settings) -> Outcome:
        if settings.master_key is None:
            return Allow(NoMasterKey(token=facts.token))
        if facts.token is None:
            return Deny(DenyReason.MISSING_KEY)
        if facts.token == "":
            return Deny(DenyReason.MALFORMED_KEY)
        if facts.route == _USER_AUTH_ROUTE:
            return Allow(UserAuthRoute()) if settings.allow_user_auth else Deny(DenyReason.USER_AUTH_DISABLED)
        if is_master_key(facts.token, settings.master_key):
            return Allow(MasterKey())
        return Defer()
