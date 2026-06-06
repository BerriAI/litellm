from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Optional

from starlette.requests import Request

from .principal import Principal

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth


class AuthMethod(str, Enum):
    """How a request authenticated. Recorded for telemetry and debugging."""

    VIRTUAL_KEY = "virtual_key"
    MASTER_KEY = "master_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    ANONYMOUS = "anonymous"


@dataclass(frozen=True)
class RequestAuthContext:
    """The single, typed auth contract every downstream stage reads.

    auth_v2 populates this once, after authentication and authorization, and
    stores it on ``request.state``. Budget/limit hooks, the end-user resolver, and
    the telemetry span all read it instead of reaching into ``request.state`` with
    untyped attribute access. Frozen so a stage can't mutate another stage's view;
    :func:`attach_end_user` produces a replaced copy.
    """

    identity: "UserAPIKeyAuth"
    principal: Principal
    auth_method: AuthMethod
    route: str
    end_user_id: Optional[str] = None


_STATE_ATTR = "auth_v2_context"


def set_auth_context(request: Request, context: RequestAuthContext) -> None:
    setattr(request.state, _STATE_ATTR, context)


def get_auth_context(request: Request) -> RequestAuthContext:
    """Return the context, or raise if auth hasn't run for this request."""
    context: Optional[RequestAuthContext] = getattr(request.state, _STATE_ATTR, None)
    if context is None:
        raise LookupError("auth_v2 context is not set on this request")
    return context


def try_get_auth_context(request: Request) -> Optional[RequestAuthContext]:
    """Return the context, or None when auth_v2 did not run (e.g. v1 path)."""
    return getattr(request.state, _STATE_ATTR, None)


def attach_end_user(request: Request, end_user_id: Optional[str]) -> RequestAuthContext:
    """Record the resolved end-user on the context (the end-user stage's output)."""
    updated = replace(get_auth_context(request), end_user_id=end_user_id)
    set_auth_context(request, updated)
    return updated
