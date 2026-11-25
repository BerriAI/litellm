from collections.abc import AsyncIterator, Callable
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.proxy.guardrails.guardrail_hooks.grayswan.grayswan import GraySwanGuardrail
from litellm.types.guardrails import GuardrailEventHooks


@pytest.fixture
async def grayswan_guardrail_factory() -> AsyncIterator[tuple[Callable[[bool, str, str], GraySwanGuardrail], Mock]]:
    respond: Final = Mock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        def make_guardrail(fail_open: bool, action: str, api_key: str) -> GraySwanGuardrail:
            guardrail: Final = GraySwanGuardrail(
                guardrail_name="failure-contract",
                api_key=api_key,
                policy_id="test-policy",
                fail_open=fail_open,
                on_flagged_action=action,
                event_hook=GuardrailEventHooks.pre_call,
            )
            guardrail.async_handler = client
            return guardrail

        yield make_guardrail, respond
