"""Pass-Through Endpoint guardrail translation handler."""

from litellm.llms.pass_through.guardrail_translation.handler import (
    PassThroughEndpointHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.pass_through: PassThroughEndpointHandler,
}

__all__ = [
    "guardrail_translation_mappings",
    "PassThroughEndpointHandler",
]
