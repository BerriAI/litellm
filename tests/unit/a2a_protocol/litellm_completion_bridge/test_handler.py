import asyncio
import datetime
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
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse

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
        (
            {"custom_llm_provider": "databricks_agent", "model": "databricks_agent/my-agent"},
            "databricks_agent/my-agent",
        ),
        ({"custom_llm_provider": "databricks_agent"}, "databricks_agent/"),
        ({"custom_llm_provider": "databricks_agent", "model": None}, "databricks_agent/"),
        ({"custom_llm_provider": "databricks_agent", "model": ""}, "databricks_agent/"),
        ({"custom_llm_provider": "langgraph"}, "langgraph/agent"),
        ({"custom_llm_provider": "langgraph", "model": None}, "langgraph/agent"),
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


def test_agent_completion_kwargs_never_expands_proxy_env_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_ONLY_SECRET", "proxy-secret-value")
    kwargs = agent_completion_kwargs(
        {
            "custom_llm_provider": "langgraph",
            "api_base": "https://agent.example.com",
            "api_key": "os.environ/PROXY_ONLY_SECRET",
        }
    )
    assert dict(kwargs) == {"api_base": "https://agent.example.com", "api_key": "os.environ/PROXY_ONLY_SECRET"}


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
async def test_databricks_app_credential_wins_over_a_forwarded_client_authorization(httpx_transport: None) -> None:
    route = respx.post(APP_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id="req-1",
        params=_a2a_params(),
        litellm_params={"custom_llm_provider": "databricks_agent", "api_key": "minted-oauth"},
        api_base=APP_URL,
        agent_extra_headers={"authorization": "Bearer client-token", "X-LiteLLM-User-Id": "u-1"},
    )
    sent = route.calls.last.request
    assert sent.headers.get_list("Authorization") == ["Bearer minted-oauth"]
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


class _CallCostCapture(CustomLogger):
    def __init__(self, logged: asyncio.Future[tuple[object, ModelResponse]]) -> None:
        super().__init__()
        self._logged = logged

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: ModelResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self._logged.set_result((kwargs.get("response_cost"), response_obj))


@respx.mock
async def test_priced_agent_call_reports_the_agents_token_price_to_callbacks(
    httpx_transport: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx.post(INVOCATIONS_URL).mock(
        return_value=httpx.Response(200, json={"object": "response", "output": [MESSAGE_ITEM]})
    )
    logged: Final[asyncio.Future[tuple[object, ModelResponse]]] = asyncio.get_running_loop().create_future()
    capture: Final = _CallCostCapture(logged)
    monkeypatch.setattr(litellm, "callbacks", [capture])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])

    await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id="req-1",
        params=_a2a_params(),
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
            "input_cost_per_token": 0.0001,
            "output_cost_per_token": 0.0002,
        },
    )
    cost, response = await asyncio.wait_for(logged, timeout=5)

    usage: Final = response.model_dump()["usage"]
    assert usage["prompt_tokens"] > 0
    assert cost == pytest.approx(usage["prompt_tokens"] * 0.0001 + usage["completion_tokens"] * 0.0002)
    assert litellm.model_cost.get("databricks_agent/my-agent", {}).get("input_cost_per_token") != 0.0001


@respx.mock
async def test_agent_with_a_price_that_is_not_a_number_answers_and_bills_its_other_price(
    httpx_transport: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx.post(INVOCATIONS_URL).mock(
        return_value=httpx.Response(200, json={"object": "response", "output": [MESSAGE_ITEM]})
    )
    logged: Final[asyncio.Future[tuple[object, ModelResponse]]] = asyncio.get_running_loop().create_future()
    capture: Final = _CallCostCapture(logged)
    monkeypatch.setattr(litellm, "callbacks", [capture])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])

    response: Final = await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id="req-1",
        params=_a2a_params(),
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
            "input_cost_per_token": "abc",
            "output_cost_per_token": 0.0002,
        },
    )
    cost, logged_response = await asyncio.wait_for(logged, timeout=5)

    assert response["result"]["parts"] == [{"kind": "text", "text": "pong"}]
    completion_tokens: Final = logged_response.model_dump()["usage"]["completion_tokens"]
    assert completion_tokens > 0
    assert cost == pytest.approx(completion_tokens * 0.0002)
