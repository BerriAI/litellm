import json
import re
from typing import Final
from unittest.mock import ANY

import httpx
import pytest
import respx

import litellm
from litellm.a2a_protocol.providers.watsonx_orchestrate.config import WatsonxOrchestrateA2AConfig
from litellm.a2a_protocol.providers.watsonx_orchestrate.handler import WXOLitellmParams

MISSING_LITELLM_PARAMS: Final = re.escape(
    "litellm_params is required for WatsonxOrchestrateA2AConfig "
    "(must contain cp4d_host, instance_id, wxo_agent_id, api_key)"
)
MISSING_HOST: Final = re.escape("'cp4d_host' is required in litellm_params for WXO agents")
RUNS_URL: Final = "https://wxo-config.test/orchestrate/cpd/instances/inst-1/v1/orchestrate/runs"
LITELLM_PARAMS: Final[WXOLitellmParams] = {
    "cp4d_host": "https://wxo-config.test",
    "instance_id": "inst-1",
    "wxo_agent_id": "agent-1",
    "api_key": "config-test-key",
    "auth_mode": "ibm_cloud",
}
EXPECTED_RUN_BODY: Final = {
    "agent_id": "agent-1",
    "message": {"role": "user", "content": [{"response_type": "text", "text": "hello"}]},
}


def _a2a_params() -> dict[str, object]:
    return {"message": {"role": "user", "parts": [{"kind": "text", "text": "hello"}], "messageId": "m-1"}}


def _mock_wxo_route(respx_mock: respx.MockRouter, url: str) -> respx.Route:
    respx_mock.post("https://iam.cloud.ibm.com/identity/token").mock(
        return_value=httpx.Response(200, json={"access_token": "tok", "expires_in": 0})
    )
    return respx_mock.post(url).mock(
        return_value=httpx.Response(200, json={"status": "completed", "results": "wxo says hi"})
    )


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@pytest.mark.parametrize("litellm_params", [None, {}])
async def test_handle_non_streaming_rejects_empty_litellm_params(litellm_params: WXOLitellmParams | None) -> None:
    with pytest.raises(ValueError, match=MISSING_LITELLM_PARAMS):
        await WatsonxOrchestrateA2AConfig().handle_non_streaming("req-1", _a2a_params(), litellm_params=litellm_params)


async def test_handle_non_streaming_requires_litellm_params_when_only_other_keywords_are_given() -> None:
    with pytest.raises(ValueError, match=MISSING_LITELLM_PARAMS):
        await WatsonxOrchestrateA2AConfig().handle_non_streaming(
            "req-1", _a2a_params(), "https://ignored.test", agent_extra_headers={"x-tenant-id": "acme"}
        )


async def test_handle_non_streaming_hands_litellm_params_to_the_wxo_handler() -> None:
    with pytest.raises(ValueError, match=MISSING_HOST):
        await WatsonxOrchestrateA2AConfig().handle_non_streaming(
            "req-1", _a2a_params(), litellm_params={"instance_id": "inst-1"}
        )


@pytest.mark.usefixtures("httpx_transport")
async def test_handle_non_streaming_runs_the_agent_and_ignores_unrelated_keywords(
    respx_mock: respx.MockRouter,
) -> None:
    runs_route: Final = _mock_wxo_route(respx_mock, RUNS_URL)

    response: Final = await WatsonxOrchestrateA2AConfig().handle_non_streaming(
        request_id="req-1",
        params=_a2a_params(),
        api_base="https://ignored.test",
        litellm_params=LITELLM_PARAMS,
        agent_extra_headers={"x-tenant-id": "acme"},
    )

    run_request: Final = runs_route.calls.last.request
    assert response == {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {
            "kind": "message",
            "role": "agent",
            "parts": [{"kind": "text", "text": "wxo says hi"}],
            "messageId": ANY,
        },
    }
    assert json.loads(run_request.content) == EXPECTED_RUN_BODY
    assert run_request.headers["authorization"] == "Bearer tok"
    assert "x-tenant-id" not in run_request.headers


@pytest.mark.parametrize("litellm_params", [None, {}])
async def test_handle_streaming_rejects_empty_litellm_params(litellm_params: WXOLitellmParams | None) -> None:
    stream: Final = WatsonxOrchestrateA2AConfig().handle_streaming(
        "req-1", _a2a_params(), litellm_params=litellm_params
    )

    with pytest.raises(ValueError, match=MISSING_LITELLM_PARAMS):
        await anext(stream)


async def test_handle_streaming_requires_litellm_params_when_only_other_keywords_are_given() -> None:
    stream: Final = WatsonxOrchestrateA2AConfig().handle_streaming(
        "req-1", _a2a_params(), "https://ignored.test", agent_extra_headers={"x-tenant-id": "acme"}
    )

    with pytest.raises(ValueError, match=MISSING_LITELLM_PARAMS):
        await anext(stream)


async def test_handle_streaming_hands_litellm_params_to_the_wxo_handler() -> None:
    stream: Final = WatsonxOrchestrateA2AConfig().handle_streaming(
        "req-1", _a2a_params(), litellm_params={"instance_id": "inst-1"}
    )

    with pytest.raises(ValueError, match=MISSING_HOST):
        await anext(stream)


@pytest.mark.usefixtures("httpx_transport")
async def test_handle_streaming_runs_the_agent_and_ignores_unrelated_keywords(respx_mock: respx.MockRouter) -> None:
    stream_route: Final = _mock_wxo_route(respx_mock, f"{RUNS_URL}/stream")

    chunks: Final = [
        chunk
        async for chunk in WatsonxOrchestrateA2AConfig().handle_streaming(
            request_id="req-1",
            params=_a2a_params(),
            api_base="https://ignored.test",
            litellm_params=LITELLM_PARAMS,
            agent_extra_headers={"x-tenant-id": "acme"},
        )
    ]

    stream_request: Final = stream_route.calls.last.request
    assert [chunk["id"] for chunk in chunks] == ["req-1", "req-1", "req-1", "req-1"]
    assert chunks[2]["result"] == {
        "contextId": ANY,
        "kind": "artifact-update",
        "taskId": ANY,
        "artifact": {"artifactId": ANY, "parts": [{"kind": "text", "text": "wxo says hi"}]},
    }
    assert chunks[3]["result"] == {
        "contextId": ANY,
        "final": True,
        "kind": "status-update",
        "status": {"state": "completed"},
        "taskId": ANY,
    }
    assert json.loads(stream_request.content) == EXPECTED_RUN_BODY
    assert stream_request.headers["authorization"] == "Bearer tok"
    assert "x-tenant-id" not in stream_request.headers
