import json

import httpx
import pytest
import respx
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.aporia_ai.aporia_ai import AporiaGuardrail

pytestmark = pytest.mark.usefixtures("httpx_transport")

_API_BASE = "https://aporia.example.test/project-1"
_USER_MESSAGE = {"role": "user", "content": "what is my balance"}


def _guardrail() -> AporiaGuardrail:
    return AporiaGuardrail(
        api_key="aporia-key",
        api_base=_API_BASE,
        guardrail_name="aporia-guard",
        event_hook="during_call",
        default_on=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("new_messages", "response_string", "expected"),
    [
        pytest.param(
            [_USER_MESSAGE],
            "your balance is 10",
            {"messages": [_USER_MESSAGE], "response": "your balance is 10", "validation_target": "both"},
            id="prompt-and-response",
        ),
        pytest.param(
            [_USER_MESSAGE],
            None,
            {"messages": [_USER_MESSAGE], "validation_target": "prompt"},
            id="prompt-only",
        ),
        pytest.param(
            [],
            "your balance is 10",
            {"messages": [], "response": "your balance is 10", "validation_target": "response"},
            id="response-only",
        ),
        pytest.param([], None, {"messages": []}, id="nothing-to-validate"),
    ],
)
async def test_prepare_aporia_request_sets_validation_target_from_supplied_content(
    new_messages: list[dict], response_string: str | None, expected: dict[str, object]
) -> None:
    request = await _guardrail().prepare_aporia_request(new_messages=new_messages, response_string=response_string)

    assert request == expected


@pytest.mark.asyncio
async def test_make_aporia_api_request_posts_prepared_body_to_validate_endpoint(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{_API_BASE}/validate").respond(json={"action": "passthrough"})

    await _guardrail().make_aporia_api_request(
        request_data={}, new_messages=[_USER_MESSAGE], response_string="your balance is 10"
    )

    sent = route.calls.last.request
    assert json.loads(sent.content) == {
        "messages": [_USER_MESSAGE],
        "response": "your balance is 10",
        "validation_target": "both",
    }
    assert sent.headers["X-APORIA-API-KEY"] == "aporia-key"


@pytest.mark.asyncio
async def test_make_aporia_api_request_raises_400_when_aporia_blocks(respx_mock: respx.MockRouter) -> None:
    verdict = {"action": "block", "revised_response": "blocked by policy"}
    respx_mock.post(f"{_API_BASE}/validate").mock(return_value=httpx.Response(200, json=verdict))

    with pytest.raises(HTTPException) as blocked:
        await _guardrail().make_aporia_api_request(
            request_data={}, new_messages=[_USER_MESSAGE], response_string="your balance is 10"
        )

    assert blocked.value.status_code == 400
    assert blocked.value.detail == {"error": "Violated guardrail policy", "aporia_ai_response": verdict}
