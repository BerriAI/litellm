from contextvars import ContextVar
from typing import Final

from litellm.proxy._types import UserAPIKeyAuth

REQUEST_AUTH: Final[ContextVar[UserAPIKeyAuth | None]] = ContextVar("litellm_pre_call_request_auth", default=None)

__all__ = ["REQUEST_AUTH"]
