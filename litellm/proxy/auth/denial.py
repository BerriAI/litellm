"""Turns a `RustDenial` into the public error the proxy raised for it before the core moved to Rust."""

from __future__ import annotations

from typing import Final, NoReturn

from fastapi import HTTPException

import litellm
from litellm.constants import INVALID_VIRTUAL_KEY_ERROR_MARKER
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_checks import TeamNotFoundError
from litellm.proxy.auth.model_access_denied import ModelAccessDeniedHTTPException
from litellm.rust_bridge._native import (
    RustAuthOutcome,
    RustDenial,
)


def _invalid_virtual_key(status: int, detail: str) -> HTTPException:
    error: Final = HTTPException(status_code=status, detail=detail)
    setattr(error, INVALID_VIRTUAL_KEY_ERROR_MARKER, True)
    return error


def denial_error(denial: RustDenial) -> Exception:
    match denial:
        case RustDenial.Proxy(status=status, message=message, error_type=error_type, param=param):
            return ProxyException(message=message, type=ProxyErrorTypes(error_type), param=param, code=status)
        case RustDenial.Http(status=status, detail=detail):
            return HTTPException(status_code=status, detail=detail)
        case RustDenial.InvalidVirtualKey(status=status, detail=detail):
            return _invalid_virtual_key(status, detail)
        case RustDenial.ModelAccess(status=status, detail=detail, internal_message=internal_message):
            return ModelAccessDeniedHTTPException(internal_message=internal_message, status_code=status, detail=detail)
        case RustDenial.Plain(message=message):
            return Exception(message)
        case RustDenial.BudgetExceeded(
            current_cost=current_cost,
            max_budget=max_budget,
            message=message,
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
        case RustDenial.TeamNotFound(team_id=team_id):
            return TeamNotFoundError(team_id=team_id)
        case unknown:
            raise RuntimeError(f"rust auth returned a denial this host does not know: {unknown!r}")


def raise_denial(outcome: RustAuthOutcome) -> NoReturn:
    """The one place a denial becomes the proxy's public error, with today's messages and statuses."""
    denial: Final = outcome.denial
    if denial is None:
        raise RuntimeError("rust auth reported a denial without one")
    raise denial_error(denial)
