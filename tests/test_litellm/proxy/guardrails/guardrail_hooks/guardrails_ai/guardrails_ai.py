from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.guardrails_ai.guardrails_ai import (
    GuardrailsAI,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.utils import Choices, Message, ModelResponse


@pytest.mark.asyncio
async def test_guardrails_ai_pre_call_hook():
    guardrails_ai_guardrail = GuardrailsAI(
        guardrail_name="guardrails_ai",
    )
