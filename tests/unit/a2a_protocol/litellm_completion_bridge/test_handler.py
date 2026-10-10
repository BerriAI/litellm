import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.a2a_protocol.litellm_completion_bridge.handler import (
    A2A_USER_API_KEY_HASH_PARAM,
    A2ACompletionBridgeHandler,
    agent_completion_kwargs,
    bridge_model_name,
)

WORKSPACE: Final = "https://adb-1.azuredatabricks.net"
INVOCATIONS_URL: Final = f"{WORKSPACE}/serving-endpoints/my-agent/invocations"
APP_URL: Final = "https://my-app-1.azure.databricksapps.com/responses"
MESSAGE_ITEM: Final = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "pong"}],
}


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def _a2a_params() -> dict[str, object]:
    return {"message": {"role": "user", "parts": [{"kind": "text", "text": "ping"}], "messageId": "m-1"}}


@pytest.mark.parametrize(
    ("litellm_params", "expected"),
    [
        ({"custom_llm_provider": "databricks_agent", "model": "my-agent"}, "databricks_agent/my-agent"),
        ({"custom_llm_provider": "databricks_agent", "model": "databricks_agent/my-agent"}, "databricks_agent/my-agent"),
        ({"custom_llm_provider": "databricks_agent"}, "databricks_agent/agent"),
        ({"custom_llm_provider": "databricks_agent", "model": None}, "databricks_agent/agent"),
        ({"custom_llm_provider": "databricks_agent", "model": ""}, "databricks_agent/"),
        ({"model": "gpt-5"}, "gpt-5"),
    ],
)
def test_bridge_model_name(litellm_params: dict[str, object], expected: str) -> None:
    assert bridge_model_name(litellm_params) == expected


def test_agent_completion_kwargs_drops_model_provider_and_agent_only_keys() -> None:
    kwargs = agent_completion_kwargs(
        {
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
            "is_public": True,
            "agent_name": "dbx",
            "agent_id": "id-1",
            "agent_card_params": {"url": WORKSPACE},
            A2A_USER_API_KEY_HASH_PARAM: "hash",
        }
    )
    assert dict(kwargs) == {"api_base": WORKSPACE, "api_key": "pat-1"}


@respx.mock
async def test_databricks_agent_message_send_through_the_bridge(httpx_transport: None) -> None:
    route = respx.post(INVOCATIONS_URL).mock(
        return_value=httpx.Response(200, json={"object": "response", "output": [MESSAGE_ITEM]})
    )
    response = await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id="req-1",
        params=_a2a_params(),
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
        },
        api_base=None,
    )
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer pat-1"
    assert json.loads(sent.content) == {"input": [{"role": "user", "content": "ping"}]}
    result = response["result"]
    assert result["kind"] == "message"
    assert result["parts"] == [{"kind": "text", "text": "pong"}]


@respx.mock
async def test_databricks_app_uses_the_minted_oauth_header_over_a_pat(httpx_transport: None) -> None:
    route = respx.post(APP_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id="req-1",
        params=_a2a_params(),
        litellm_params={"custom_llm_provider": "databricks_agent", "api_key": "pat-1"},
        api_base=APP_URL,
        agent_extra_headers={"Authorization": "Bearer minted-oauth", "X-LiteLLM-User-Id": "u-1"},
    )
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer minted-oauth"
    assert sent.headers["X-LiteLLM-User-Id"] == "u-1"


@respx.mock
async def test_databricks_agent_message_stream_through_the_bridge(httpx_transport: None) -> None:
    events = [
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "po"},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "ng"},
        {"type": "response.output_item.done", "item": MESSAGE_ITEM},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    respx.post(INVOCATIONS_URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
    )
    chunks = [
        chunk
        async for chunk in A2ACompletionBridgeHandler.handle_streaming(
            request_id="req-1",
            params=_a2a_params(),
            litellm_params={
                "custom_llm_provider": "databricks_agent",
                "model": "my-agent",
                "api_base": WORKSPACE,
                "api_key": "pat-1",
            },
        )
    ]
    kinds = [chunk["result"]["kind"] for chunk in chunks]
    assert kinds == ["task", "status-update", "artifact-update", "status-update"]
    assert chunks[2]["result"]["artifact"]["parts"] == [{"kind": "text", "text": "pong"}]
    assert chunks[3]["result"]["status"]["state"] == "completed"
