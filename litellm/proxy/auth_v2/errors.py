from __future__ import annotations

from typing import Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse

SCIM_ERROR_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:Error"


class AuthError(HTTPException):
    def __init__(
        self, status_code: int, detail: str, challenge: Optional[str] = None
    ) -> None:
        headers = {"WWW-Authenticate": challenge} if challenge else None
        super().__init__(status_code=status_code, detail=detail, headers=headers)


def bearer_challenge(
    error: Optional[str] = None, description: Optional[str] = None
) -> str:
    parts = ['Bearer realm="litellm"']
    if error:
        parts.append(f'error="{error}"')
    if description:
        parts.append(f'error_description="{description}"')
    return ", ".join(parts)


def basic_challenge(realm: str = "litellm") -> str:
    return f'Basic realm="{realm}"'


def unauthenticated(challenge: str) -> AuthError:
    return AuthError(401, "Not authenticated", challenge)


def invalid_token(description: Optional[str] = None) -> AuthError:
    return AuthError(
        401, "Invalid token", bearer_challenge("invalid_token", description)
    )


def insufficient_scope() -> AuthError:
    return AuthError(403, "Insufficient scope", bearer_challenge("insufficient_scope"))


def forbidden_role() -> AuthError:
    return AuthError(403, "Insufficient role")


def forbidden_permission() -> AuthError:
    return AuthError(403, "Forbidden")


def account_disabled() -> AuthError:
    return AuthError(403, "Account disabled")


def scim_error_response(exc: Exception) -> JSONResponse:
    status_code = exc.status_code if isinstance(exc, HTTPException) else 500
    detail = exc.detail if isinstance(exc, HTTPException) else "Internal server error"
    return JSONResponse(
        status_code=status_code,
        content={
            "schemas": [SCIM_ERROR_SCHEMA],
            "status": str(status_code),
            "detail": str(detail),
        },
    )
