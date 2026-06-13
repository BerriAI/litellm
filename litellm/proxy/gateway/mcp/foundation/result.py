"""Result alias for the gateway.

Re-exports the Expression Result spine and pins the error channel to
GatewayError so call sites can write GatewayResult[T] instead of
Result[T, GatewayError].
"""

from __future__ import annotations

from typing import TypeAlias, TypeVar

from expression import Error, Ok, Result

from litellm.proxy.gateway.mcp.foundation.errors import GatewayError

_T = TypeVar("_T")

GatewayResult: TypeAlias = Result[_T, GatewayError]

__all__ = ["Error", "GatewayError", "GatewayResult", "Ok", "Result"]
