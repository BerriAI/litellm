from typing import Final, TypeGuard

from fastapi import HTTPException

import litellm
from litellm.proxy._types import (
    ProxyErrorTypes,
    ProxyException,
)
from litellm.proxy.auth import denial as auth_denial
from litellm.proxy.auth.auth_checks import TeamNotFoundError
from litellm.proxy.auth.auth_utils import is_invalid_virtual_key_error
from litellm.proxy.auth.model_access_denied import ModelAccessDeniedHTTPException
from litellm.proxy.auth.resolvers.exceptions import KeyNotFoundError
from litellm.rust_bridge._native import RustDenial


def test_proxy_denial_keeps_its_status_type_and_param() -> None:
    error: Final = auth_denial.denial_error(
        RustDenial.Proxy(status=401, message="expired", error_type="expired_key", param="sk-...abcd")
    )

    assert isinstance(error, ProxyException)
    assert (error.message, error.type, error.param, error.code) == (
        "expired",
        ProxyErrorTypes.expired_key,
        "sk-...abcd",
        "401",
    )


def test_invalid_virtual_key_denial_is_marked_for_log_routing() -> None:
    error: Final = auth_denial.denial_error(
        RustDenial.InvalidVirtualKey(status=401, detail="LiteLLM Virtual Key expected.")
    )

    assert isinstance(error, HTTPException)
    assert (error.status_code, error.detail) == (401, "LiteLLM Virtual Key expected.")
    assert is_invalid_virtual_key_error(error)


def test_plain_http_denial_is_not_marked_as_an_invalid_key() -> None:
    error: Final = auth_denial.denial_error(
        RustDenial.Http(status=403, detail="'allow_user_auth' not set or set to False")
    )

    assert isinstance(error, HTTPException)
    assert error.status_code == 403
    assert not is_invalid_virtual_key_error(error)


def test_model_access_denial_carries_the_internal_message() -> None:
    error: Final = auth_denial.denial_error(
        RustDenial.ModelAccess(status=401, detail="no access", internal_message="model=x")
    )

    assert isinstance(error, ModelAccessDeniedHTTPException)
    assert (error.status_code, error.detail, error.internal_message) == (401, "no access", "model=x")


def test_plain_denial_is_a_bare_exception() -> None:
    error: Final = auth_denial.denial_error(RustDenial.Plain(message="no team"))

    assert type(error) is Exception
    assert str(error) == "no team"


def test_budget_denial_is_the_budget_error_the_host_step_raised() -> None:
    error: Final = auth_denial.denial_error(
        RustDenial.BudgetExceeded(
            current_cost=2.0, max_budget=1.0, message="over", entity_type="team_member", entity_id="u:t"
        )
    )

    assert isinstance(error, litellm.BudgetExceededError)
    assert (error.current_cost, error.max_budget, error.message, error.entity_type, error.entity_id) == (
        2.0,
        1.0,
        "over",
        "team_member",
        "u:t",
    )


def test_gone_session_team_is_a_team_not_found() -> None:
    error: Final = auth_denial.denial_error(RustDenial.TeamNotFound(team_id="t"))

    assert isinstance(error, TeamNotFoundError)
    assert error.status_code == 404


_ERROR_CLASSES: Final[dict[RustDenial, type[Exception]]] = {
    RustDenial.Proxy(status=401, message="m", error_type="auth_error", param=None): ProxyException,
    RustDenial.Http(status=403, detail="d"): HTTPException,
    RustDenial.InvalidVirtualKey(status=401, detail="d"): HTTPException,
    RustDenial.ModelAccess(status=401, detail="d", internal_message="i"): ModelAccessDeniedHTTPException,
    RustDenial.Plain(message="m"): Exception,
    RustDenial.BudgetExceeded(
        current_cost=2.0, max_budget=1.0, message="m", entity_type=None, entity_id=None
    ): litellm.BudgetExceededError,
    RustDenial.TeamNotFound(team_id="t"): TeamNotFoundError,
    RustDenial.KeyNotFound(hash="h"): KeyNotFoundError,
    RustDenial.NoMatchingJwtPublicKey(message="m"): auth_denial.NoMatchingJWTPublicKeyError,
    RustDenial.IdJagNotBearer(message="m"): auth_denial.IdJagIsNotABearerTokenError,
}


def _is_variant(value: object) -> TypeGuard[type[RustDenial]]:
    return isinstance(value, type) and issubclass(value, RustDenial)


def test_every_denial_the_core_renders_is_raised_as_exactly_one_named_class() -> None:
    raised: Final = {type(denial): type(auth_denial.denial_error(denial)) for denial in _ERROR_CLASSES}

    assert raised == {type(denial): error_class for denial, error_class in _ERROR_CLASSES.items()}
    assert set(raised) == {value for value in vars(RustDenial).values() if _is_variant(value)}
