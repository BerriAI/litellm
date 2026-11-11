"""Runs one gateway-auth-core call to its outcome, answering each step it parks on from a `PythonAuthHost`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import litellm
from litellm.proxy._types import LiteLLM_JWTAuth
from litellm.proxy.auth.host.answer import answer
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.rust_bridge._native import (
    RustAuth,
    RustAuthOutcome,
    RustAuthSettings,
    RustAuthStep,
)
from litellm.secret_managers.main import get_secret_bool

if TYPE_CHECKING:
    from litellm.models.user import LiteLLM_UserTable
    from litellm.proxy.auth.host.python import PythonAuthHost
    from litellm.proxy.auth.host.steps.authorization import Authorization
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


@dataclass(frozen=True, slots=True)
class AuthRequest:
    route: str
    method: str | None
    token: str | None
    model: str | None
    team_header: str | None
    is_llm_api_route: bool
    auth_enforced_pass_through: bool
    forwarded_routes: tuple[str, ...]
    is_info_route: bool
    now: float
    pass_through_open: bool = False
    from_custom_auth: bool = False


def build_settings(
    general_settings: Mapping[str, object],
    litellm_jwtauth: LiteLLM_JWTAuth | None,
    *,
    master_key: str | None,
    premium_user: bool,
    has_db: bool,
) -> RustAuthSettings:
    """Built by proxy_server on every config load, next to the rest of general_settings."""
    return RustAuthSettings.from_general_settings(
        general_settings,
        litellm_jwtauth,
        master_key=master_key,
        premium=premium_user,
        has_db=has_db,
        ui_login=get_secret_bool("EXPERIMENTAL_UI_LOGIN") is not False,
        proxy_budget=litellm.max_budget > 0,
        metrics_require_auth=litellm.require_auth_for_metrics_endpoint is not False,
    )


def _report(call: RustAuth, error: Exception) -> None:
    """Tells the core how a host step failed: an unreachable database, which the core serves through or
    denies, or anything else, which the core skips where the proxy tolerated it."""
    transient: Final = PrismaDBExceptionHandler.is_database_connection_error(error)
    if not transient and not PrismaDBExceptionHandler.is_database_service_unavailable_error(error):
        call.fail()
        return
    call.unavailable(
        dependency="db", transient=transient, message=PrismaDBExceptionHandler.database_unavailable_message(error)
    )


async def drive(call: RustAuth, host: PythonAuthHost) -> RustAuthOutcome:
    """A host step that raises is reported to the core, which skips a step whose failure the proxy
    tolerated and decides what an unreachable database means. When the core gives up instead, the step's
    exception propagates as the Python auth path raised it."""
    failure: Exception | None = None  # rebind-ok: the latest host failure, re-raised if the core gives up on it
    step = call.resume()  # rebind-ok: each resume returns the next step
    while not isinstance(step, RustAuthStep.Done):
        try:
            await answer(call, step, host)
        except Exception as error:
            failure = error  # rebind-ok: see above
            _report(call, error)
        try:
            step = call.resume()  # rebind-ok: each resume returns the next step
        except RuntimeError:
            if failure is None:
                raise
            raise failure
    return step.outcome


async def authenticate_key(
    request: AuthRequest, key: str | None, settings: RustAuthSettings, host: PythonAuthHost
) -> RustAuthOutcome:
    return await drive(RustAuth.authenticate_key(request, key, settings), host)


async def authenticate_jwt(request: AuthRequest, settings: RustAuthSettings, host: PythonAuthHost) -> RustAuthOutcome:
    return await drive(RustAuth.authenticate_jwt(request, settings), host)


async def authorize(authorization: Authorization) -> None:
    """`common_checks` for an authenticated request: raises the proxy's error for the first check that refuses it."""
    from litellm.proxy.auth.denial import raise_denial  # noqa: PLC0415  # denial imports auth_checks
    from litellm.proxy.auth.host.python import proxy_host  # noqa: PLC0415  # host.python imports user_api_key_auth

    outcome: Final = await drive(
        RustAuth.authorize(authorization.facts()),
        proxy_host(authorization.route, None, authorization=authorization),
    )
    if outcome.denied:
        raise_denial(outcome)


async def key_max_budget(token: UserAPIKeyAuth, owner: LiteLLM_UserTable | None) -> None:
    """The key's live spend against its max budget, for callers outside authentication."""
    from litellm.proxy.auth.denial import raise_denial  # noqa: PLC0415  # denial imports auth_checks
    from litellm.proxy.auth.host.facts import key_facts  # noqa: PLC0415  # host.facts imports auth_checks
    from litellm.proxy.auth.host.python import proxy_host  # noqa: PLC0415  # host.python imports user_api_key_auth

    outcome: Final = await drive(RustAuth.key_max_budget(key_facts(token), token, owner), proxy_host("", None))
    if outcome.denied:
        raise_denial(outcome)
