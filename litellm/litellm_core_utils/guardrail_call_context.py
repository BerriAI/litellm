from contextvars import ContextVar
from typing import Final

from litellm.types.utils import CallTypes

guardrail_call_type: Final[ContextVar[CallTypes | None]] = ContextVar("guardrail_call_type", default=None)
