import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import asyncio

import httpx
import pytest
import respx
from fastapi import Request, Response

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import assemblyai_proxy_route

_KEY: Final = "synthetic-assemblyai-key"
_TRANSCRIPT: Final = {
    "id": "tr_1",
    "status": "queued",
    "audio_url": "https://assembly.ai/wildfires.mp3",
}


def _request(method: str, url: str, body: bytes = b"{}") -> Request:
    request: Final = MagicMock(spec=Request)
    request.method = method
    request.url = httpx.URL(url)
    request.headers = {"content-type": "application/json"}
    request.scope = {"path": httpx.URL(url).path, "type": "http", "method": method, "headers": []}
    request.query_params = {}
    request.body = AsyncMock(return_value=body)
    request.json = AsyncMock(return_value=json.loads(body))
    return request


@pytest.fixture(autouse=True)
def _assemblyai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", _KEY)


@pytest.mark.asyncio
async def test_assemblyai_transcribe_create_poll_delete(respx_mock: respx.MockRouter, httpx_transport) -> None:
    create_route: Final = respx_mock.post("https://api.assemblyai.com/v2/transcript").mock(
        return_value=httpx.Response(200, json={**_TRANSCRIPT, "id": "tr_1"})
    )
    poll_route: Final = respx_mock.get("https://api.assemblyai.com/v2/transcript/tr_1").mock(
        side_effect=[
            httpx.Response(200, json={**_TRANSCRIPT, "id": "tr_1", "status": "queued"}),
            httpx.Response(200, json={**_TRANSCRIPT, "id": "tr_1", "status": "completed", "text": "fires near town"}),
        ]
    )
    delete_route: Final = respx_mock.delete("https://api.assemblyai.com/v2/transcript/tr_1").mock(
        return_value=httpx.Response(200, json={**_TRANSCRIPT, "id": "tr_1", "status": "deleted"})
    )

    user: Final = MagicMock()
    create: Final = await assemblyai_proxy_route(
        endpoint="v2/transcript",
        request=_request(
            "POST",
            "http://proxy/assemblyai/v2/transcript",
            json.dumps({"audio_url": "https://assembly.ai/wildfires.mp3", "speech_models": ["universal-2"]}).encode(),
        ),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=user,
    )
    poll: Final = await assemblyai_proxy_route(
        endpoint="v2/transcript/tr_1",
        request=_request("GET", "http://proxy/assemblyai/v2/transcript/tr_1"),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=user,
    )
    delete: Final = await assemblyai_proxy_route(
        endpoint="v2/transcript/tr_1",
        request=_request("DELETE", "http://proxy/assemblyai/v2/transcript/tr_1"),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=user,
    )

    assert create_route.call_count == 1
    outbound: Final = create_route.calls[0].request
    assert outbound.headers["authorization"] == _KEY
    sent: Final = json.loads(outbound.content)
    assert sent["audio_url"] == "https://assembly.ai/wildfires.mp3"
    assert sent["speech_models"] == ["universal-2"]
    assert json.loads(create.body)["id"] == "tr_1"
    assert poll_route.call_count == 1
    assert delete_route.call_count == 1


@pytest.mark.asyncio
async def test_assemblyai_transcribe_with_non_admin_key_logs_key_identity(
    respx_mock: respx.MockRouter, httpx_transport
) -> None:
    recorded: Final = []

    class _Recorder(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
            recorded.append(kwargs)

        def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
            recorded.append(kwargs)

    litellm._async_success_callback = [*litellm._async_success_callback, _Recorder()]
    litellm.success_callback = [*litellm.success_callback, _Recorder()]

    respx_mock.post("https://api.assemblyai.com/v2/transcript").mock(
        return_value=httpx.Response(200, json={**_TRANSCRIPT, "id": "tr_9"})
    )
    respx_mock.get("https://api.assemblyai.com/v2/transcript/tr_9").mock(
        return_value=httpx.Response(
            200, json={**_TRANSCRIPT, "id": "tr_9", "status": "completed", "text": "fires near town"}
        )
    )
    non_admin: Final = UserAPIKeyAuth(
        api_key="non-admin-virtual-key",
        token="non-admin-virtual-key",
        user_id="non-admin-user",
        user_role="internal_user",
    )
    import time as _time
    from datetime import datetime
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
    logging_obj: Final = LiteLLMLoggingObj(
        model="assemblyai", messages=[], stream=False, call_type="pass_through_endpoint",
        start_time=_time.time(), litellm_call_id="call-x", function_id="f")
    logging_obj.model_call_details["litellm_params"] = {"success_callback": [_Recorder()], "async_success_callback": [_Recorder()]}
    logging_obj.model_call_details["metadata"] = {
        "user_api_key": "non-admin-virtual-key",
        "user_api_key_user_id": "non-admin-user",
        "user_api_key_team_id": "non-admin-team",
    }
    response = await assemblyai_proxy_route(
        endpoint="v2/transcript",
        request=_request("POST", "http://proxy/assemblyai/v2/transcript", b'{"audio_url": "https://a"}'),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=non_admin)
    assert response.status_code == 200, response.body
    assert respx_mock.calls.call_count == 1
    assert respx_mock.calls.last.request.headers["authorization"] == _KEY
    respx_mock.get(url__regex=r".*transcript/tr_9.*").mock(
        return_value=httpx.Response(200, json={"id": "tr_9", "status": "queued"}))
    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=httpx.Response(200, json={"id": "tr_9"}, request=httpx.Request("POST", "https://api.assemblyai.com/v2/transcript")),
        response_body={"id": "tr_9", "status": "completed", "text": "hi"}, logging_obj=logging_obj,
        url_route="assemblyai/v2/transcript", result="", start_time=datetime.now(), end_time=datetime.now(),
        cache_hit=False, request_body={"audio_url": "https://a"},
        passthrough_logging_payload={"url": "https://api.assemblyai.com/v2/transcript", "request_method": "POST",
                                     "request_body": {"audio_url": "https://a"}, "response_body": {"id": "tr_9", "status": "completed", "text": "hi"}},
        user_api_key_dict=non_admin, user_api_key="non-admin-virtual-key", user_api_key_user_id="non-admin-user")
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    payload: Final = logging_obj.model_call_details["passthrough_logging_payload"]
    assert payload["request_method"] == "POST" and "assemblyai.com/v2/transcript" in payload["url"], payload
    metadata: Final = logging_obj.model_call_details["metadata"]
    assert metadata["user_api_key"] == "non-admin-virtual-key" and metadata["user_api_key_user_id"] == "non-admin-user"
