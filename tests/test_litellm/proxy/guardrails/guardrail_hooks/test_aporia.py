import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.guardrails.guardrail_hooks.aporia_ai.aporia_ai import AporiaGuardrail


def _guardrail(body: dict[str, str | None]) -> tuple[AporiaGuardrail, AsyncMock]:
    response = MagicMock()
    response.status_code = 200
    response.text = json.dumps(body)
    response.json = MagicMock(return_value=body)
    post = AsyncMock(return_value=response)
    handler = MagicMock(spec=AsyncHTTPHandler)
    handler.post = post
    guardrail = AporiaGuardrail(
        guardrail_name="aporia",
        api_key="k",
        api_base="https://example.invalid",
        async_handler=handler,
    )
    return guardrail, post


async def _validate(guardrail: AporiaGuardrail) -> None:
    await guardrail.make_aporia_api_request(
        request_data={},
        new_messages=[{"role": "user", "content": "hello"}],
    )


@pytest.mark.asyncio
async def test_passthrough_is_forwarded():
    guardrail, post = _guardrail({"action": "passthrough"})

    await _validate(guardrail)

    post.assert_awaited_once()


@pytest.mark.asyncio
async def test_block_is_refused():
    guardrail, _ = _guardrail({"action": "block"})

    with pytest.raises(HTTPException) as exc:
        await _validate(guardrail)

    assert exc.value.status_code == 400
    assert "Violated guardrail policy" in str(exc.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["modify", "rephrase"])
async def test_an_intervention_is_not_forwarded_unchanged(action: str):
    guardrail, _ = _guardrail({"action": action})

    with pytest.raises(HTTPException) as exc:
        await _validate(guardrail)

    assert exc.value.status_code == 400
    assert action in str(exc.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [{"action": "BLOCK"}, {"action": "blocked"}, {"action": ""}, {"action": None}, {}],
    ids=["BLOCK", "blocked", "empty", "null", "missing"],
)
async def test_a_verdict_it_cannot_read_is_not_taken_for_permission(body: dict[str, str | None]):
    guardrail, _ = _guardrail(body)

    with pytest.raises(HTTPException):
        await _validate(guardrail)
