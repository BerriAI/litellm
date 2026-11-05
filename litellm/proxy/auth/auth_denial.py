from typing import assert_never

from fastapi import HTTPException, status

from litellm._auth.decision import DenyReason
from litellm.proxy._types import ProxyErrorTypes, ProxyException


def denial_exception(reason: DenyReason) -> ProxyException | HTTPException:
    match reason:
        case DenyReason.MISSING_KEY:
            return ProxyException(
                message="Authentication Error, No api key passed in.",
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            )
        case DenyReason.MALFORMED_KEY:
            return ProxyException(
                message="Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix.",
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            )
        case DenyReason.USER_AUTH_DISABLED:
            return HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="'allow_user_auth' not set or set to False",
            )
        case _:
            assert_never(reason)
