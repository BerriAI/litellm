"""
Pydantic AI agent provider for A2A protocol.

Pydantic AI agents follow A2A protocol but don't support streaming natively.
This provider handles fake streaming by converting non-streaming responses into streaming chunks.
"""

from litellm.a2a_protocol.providers.pydantic_ai_agents.config import (
    PydanticAIProviderConfig,
)
from litellm.a2a_protocol.providers.pydantic_ai_agents.handler import PydanticAIHandler
from litellm.a2a_protocol.providers.pydantic_ai_agents.transformation import (
    PydanticAITransformation,
)

__all__ = ["PydanticAIHandler", "PydanticAITransformation", "PydanticAIProviderConfig"]

