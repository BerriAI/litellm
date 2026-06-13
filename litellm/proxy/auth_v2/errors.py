from __future__ import annotations

from typing import Optional

from fastapi import HTTPException


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


def unknown_provider() -> HTTPException:
    return HTTPException(status_code=404, detail="unknown provider")


def invalid_login_state() -> HTTPException:
    return HTTPException(status_code=400, detail="invalid or expired login state")


def state_mismatch() -> HTTPException:
    return HTTPException(status_code=400, detail="state mismatch")


def oidc_provider_error(error: str) -> HTTPException:
    return HTTPException(status_code=400, detail=error)


def missing_authorization_code() -> HTTPException:
    return HTTPException(status_code=400, detail="missing authorization code")


def saml_redirect_failed() -> HTTPException:
    return HTTPException(status_code=500, detail="no SAML redirect produced")


def missing_saml_response() -> HTTPException:
    return HTTPException(status_code=400, detail="missing SAMLResponse")


def invalid_saml_response() -> HTTPException:
    return HTTPException(status_code=401, detail="invalid SAML response")


def saml_assertion_replay() -> HTTPException:
    return HTTPException(status_code=401, detail="SAML assertion replay")
