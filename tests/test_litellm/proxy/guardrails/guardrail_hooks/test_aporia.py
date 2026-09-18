"""Tests for how the Aporia guardrail acts on Aporia's verdict.

Aporia answers with one of four actions. Only ``passthrough`` says the content
may go out as written; ``modify`` and ``rephrase`` are interventions. Forwarding
the original for those defeats the guardrail and reports success while doing it
(#41097).
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.aporia_ai.aporia_ai import AporiaGuardrail


def _guardrail(action: Any, include_action: bool = True) -> AporiaGuardrail:
    """An AporiaGuardrail whose /validate call answers with one verdict."""
    guardrail = AporiaGuardrail(
        guardrail_name="aporia",
        api_key="k",
        api_base="https://example.invalid",
    )
    body = {"action": action} if include_action else {"reason": "no action key"}
    response = MagicMock()
    response.status_code = 200
    response.text = json.dumps(body)
    response.json = MagicMock(return_value=body)
    guardrail.async_handler = MagicMock()
    guardrail.async_handler.post = AsyncMock(return_value=response)
    return guardrail


async def _validate(guardrail: AporiaGuardrail) -> None:
    await guardrail.make_aporia_api_request(
        request_data={},
        new_messages=[{"role": "user", "content": "hello"}],
    )


@pytest.mark.asyncio
async def test_passthrough_is_forwarded():
    """The accept control: a cleanly scanned request must still go out, or the
    guardrail is a wall rather than a filter."""
    guardrail = _guardrail("passthrough")

    await _validate(guardrail)

    # It reached Aporia and came back without raising, which is the whole
    # observable effect of letting a request through.
    guardrail.async_handler.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_block_is_refused():
    with pytest.raises(HTTPException) as exc:
        await _validate(_guardrail("block"))

    assert exc.value.status_code == 400
    assert "Violated guardrail policy" in str(exc.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["modify", "rephrase"])
async def test_an_intervention_is_not_forwarded_unchanged(action):
    """Aporia is saying this should not ship as written, and litellm cannot
    apply the modification from this response — so it must not ship it."""
    with pytest.raises(HTTPException) as exc:
        await _validate(_guardrail(action))

    assert exc.value.status_code == 400
    assert action in str(exc.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,include_action",
    [
        ("BLOCK", True),
        ("blocked", True),
        ("", True),
        (None, True),
        (None, False),
    ],
)
async def test_a_verdict_it_cannot_read_is_not_taken_for_permission(action, include_action):
    """A response shape change, or an error object with no action at all, used
    to forward. A guardrail that cannot tell what it was told is not the one to
    decide the content is fine."""
    with pytest.raises(HTTPException):
        await _validate(_guardrail(action, include_action=include_action))
