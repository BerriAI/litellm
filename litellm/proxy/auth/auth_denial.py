from typing import Final, assert_never

from fastapi import HTTPException, status

import litellm
from litellm.constants import INVALID_VIRTUAL_KEY_ERROR_MARKER, INVALID_VIRTUAL_KEY_ERROR_MESSAGE
from litellm.proxy._types import Litellm_EntityType, ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import key_budget_exceeded_error
from litellm.proxy.auth.auth_utils import abbreviate_api_key
from litellm.proxy.auth.core.identity import (
    ExpiredKey,
    KeyBlocked,
    KeyDenial,
    KeyWithoutToken,
    MalformedKey,
    MissingKey,
    NoDatabase,
    NotAVirtualKey,
    OwnerDeactivated,
)
from litellm.proxy.auth.core.outcome import (
    Denial,
    KeyBudgetExceeded,
    MasterKeyOnlyRoute,
    ModelBudgetExceeded,
    TeamMemberBudgetExceeded,
    UserAuthDisabled,
)
from litellm.types.proxy.auth.user_api_key_auth import is_jwt

_JWT_AUTH_DISABLED_HINT: Final = (
    " This key has the structure of a JWT, but JWT auth is not enabled on this proxy, so it was treated as a"
    " virtual key. Set `enable_jwt_auth: true` under `general_settings` in your proxy config to authenticate"
    " with JWTs."
)


def _not_a_virtual_key(token: str, jwt_auth_enabled: bool) -> HTTPException:
    masked: Final = f"{token[:4]}****{token[-4:]}" if len(token) > 8 else "****"
    hint: Final = _JWT_AUTH_DISABLED_HINT if not jwt_auth_enabled and is_jwt(token=token) else ""
    error: Final = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"{INVALID_VIRTUAL_KEY_ERROR_MESSAGE}. Received={masked}, expected to start with 'sk-'.{hint}",
    )
    # Log routing classifies this 401 by where it was raised, never by its message text.
    setattr(error, INVALID_VIRTUAL_KEY_ERROR_MARKER, True)
    return error


def denial_exception(
    denial: KeyDenial | Denial,
    token: str | None,
    route: str,
    jwt_auth_enabled: bool,
    key: UserAPIKeyAuth | None,
) -> ProxyException | HTTPException | litellm.BudgetExceededError:
    match denial:
        case MissingKey():
            return ProxyException(
                message="Authentication Error, No api key passed in.",
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            )
        case MalformedKey():
            return ProxyException(
                message="Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix.",
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            )
        case UserAuthDisabled():
            return HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="'allow_user_auth' not set or set to False",
            )
        case ExpiredKey(expires=expires, now=now, key_label=key_label):
            return ProxyException(
                message=f"Authentication Error - Expired Key. Key Expiry time {expires} and current time {now}",
                type=ProxyErrorTypes.expired_key,
                code=status.HTTP_401_UNAUTHORIZED,
                param=abbreviate_api_key(api_key=key_label),
            )
        case MasterKeyOnlyRoute():
            return ProxyException(
                message=f"Authentication Error, Tried to access route={route}, which is only for MASTER KEY",
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            )
        case NoDatabase():
            return ProxyException(
                message="No connected db.",
                type=ProxyErrorTypes.no_db_connection,
                code=400,
                param=None,
            )
        case NotAVirtualKey():
            return _not_a_virtual_key(token or "", jwt_auth_enabled)
        case KeyBlocked():
            return ProxyException(
                message="Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin.",
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            )
        case OwnerDeactivated(user_id=user_id):
            return ProxyException(
                message=(
                    f"Authentication Error, User={user_id} has been deactivated via SCIM. "
                    "Keys owned by this user cannot be used."
                ),
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            )
        case TeamMemberBudgetExceeded(user_id=user_id, team_id=team_id, spend=spend, max_budget=max_budget):
            return litellm.BudgetExceededError(
                current_cost=spend,
                max_budget=max_budget,
                message=(
                    f"Budget has been exceeded! TeamMember={user_id}:{team_id} "
                    f"Current cost: {spend}, Max budget: {max_budget}"
                ),
                entity_type=Litellm_EntityType.TEAM_MEMBER.value,
                entity_id=f"{user_id}:{team_id}",
            )
        case KeyBudgetExceeded(spend=spend, max_budget=max_budget):
            return key_budget_exceeded_error(key or UserAPIKeyAuth(), spend, max_budget)
        case KeyWithoutToken():
            return HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key, no token associated"
            )
        case ModelBudgetExceeded(
            message=message,
            current_cost=current_cost,
            max_budget=max_budget,
            entity_type=entity_type,
            entity_id=entity_id,
        ):
            return litellm.BudgetExceededError(
                current_cost=current_cost,
                max_budget=max_budget,
                message=message,
                entity_type=entity_type,
                entity_id=entity_id,
            )
        case _:
            assert_never(denial)
