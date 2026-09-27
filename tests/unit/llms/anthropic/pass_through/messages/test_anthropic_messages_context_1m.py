import json
from typing import Final

import pytest
import respx

import litellm
from litellm import Router


@pytest.mark.parametrize(
    "deployment_model,expected_betas",
    [
        ("anthropic/claude-sonnet-4-6[1m]", {"context-1m-2025-08-07", "interleaved-thinking-2025-05-14"}),
        ("anthropic/claude-sonnet-4-6", {"interleaved-thinking-2025-05-14"}),
    ],
)
async def test_router_messages_context_1m_deployment_sends_base_model_with_beta(
    deployment_model: str,
    expected_betas: set[str],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    monkeypatch.setattr(litellm.anthropic_beta_headers_manager, "_BETA_HEADERS_CONFIG", None)
    route: Final = respx_mock.post("https://api.anthropic.com/v1/messages").respond(
        200,
        json={
            "id": "msg_context_1m",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4-6[1m]",
                "litellm_params": {"model": deployment_model, "api_key": "test"},
            }
        ]
    )

    await router.aanthropic_messages(
        model="claude-sonnet-4-6[1m]",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16,
        extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
    )

    request: Final = route.calls.last.request
    assert json.loads(request.content)["model"] == "claude-sonnet-4-6"
    assert {beta.strip() for beta in request.headers["anthropic-beta"].split(",")} == expected_betas
