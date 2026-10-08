import asyncio
import json
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Final, cast

import httpx
import pytest
import respx
from fastapi import Request, Response
from starlette.types import Message

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import assemblyai_proxy_route
from litellm.types.utils import StandardLoggingPayload

_KEY: Final = "synthetic-assemblyai-key"
_UPSTREAM: Final = "https://api.assemblyai.com/v2/transcript"
_AUDIO_URL: Final = "https://assembly.ai/wildfires.mp3"


class _SuccessRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: Final[list[StandardLoggingPayload]] = []

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.payloads.append(cast(StandardLoggingPayload, kwargs["standard_logging_object"]))


def _request(method: str, path: str, body: bytes = b"") -> Request:
    scope: Final = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 51234),
        "server": ("proxy.local", 4000),
        "state": {},
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _transcript_logs(recorder: _SuccessRecorder, transcript_id: str) -> tuple[StandardLoggingPayload, ...]:
    return tuple(
        payload
        for payload in recorder.payloads
        if isinstance(payload["response"], dict) and payload["response"].get("id") == transcript_id
    )


async def _wait_for_success_log(recorder: _SuccessRecorder, transcript_id: str) -> StandardLoggingPayload:
    deadline: Final = time.monotonic() + 10
    while not _transcript_logs(recorder, transcript_id) and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    logged: Final = _transcript_logs(recorder, transcript_id)
    assert len(logged) == 1, f"AssemblyAI success log for {transcript_id} emitted {len(logged)} times"
    return logged[0]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _SuccessRecorder:
    success_recorder: Final = _SuccessRecorder()
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", _KEY)
    monkeypatch.setattr(litellm, "_async_success_callback", [success_recorder])
    monkeypatch.setattr(litellm, "success_callback", [])
    return success_recorder


@pytest.mark.asyncio
async def test_assemblyai_transcribe_create_poll_delete(
    respx_mock: respx.MockRouter, httpx_transport: None, recorder: _SuccessRecorder
) -> None:
    create_route: Final = respx_mock.post(_UPSTREAM).mock(
        return_value=httpx.Response(200, json={"id": "tr_1", "status": "queued", "audio_url": _AUDIO_URL})
    )
    poll_route: Final = respx_mock.get(f"{_UPSTREAM}/tr_1").mock(
        return_value=httpx.Response(200, json={"id": "tr_1", "status": "completed", "text": "fires near town"})
    )
    delete_route: Final = respx_mock.delete(f"{_UPSTREAM}/tr_1").mock(
        return_value=httpx.Response(200, json={"id": "tr_1", "status": "deleted"})
    )
    admin: Final = UserAPIKeyAuth(api_key="sk-master", user_role=LitellmUserRoles.PROXY_ADMIN)
    create_body: Final = {"audio_url": _AUDIO_URL, "speech_models": ["universal-2"]}

    create: Final = await assemblyai_proxy_route(
        endpoint="v2/transcript",
        request=_request("POST", "/assemblyai/v2/transcript", json.dumps(create_body).encode()),
        fastapi_response=Response(),
        user_api_key_dict=admin,
    )
    await _wait_for_success_log(recorder, "tr_1")
    poll: Final = await assemblyai_proxy_route(
        endpoint="v2/transcript/tr_1",
        request=_request("GET", "/assemblyai/v2/transcript/tr_1"),
        fastapi_response=Response(),
        user_api_key_dict=admin,
    )
    delete: Final = await assemblyai_proxy_route(
        endpoint="v2/transcript/tr_1",
        request=_request("DELETE", "/assemblyai/v2/transcript/tr_1"),
        fastapi_response=Response(),
        user_api_key_dict=admin,
    )

    assert isinstance(create, Response) and isinstance(poll, Response) and isinstance(delete, Response)
    assert create.status_code == 200
    assert json.loads(create.body)["id"] == "tr_1"
    assert poll.status_code == 200
    assert json.loads(poll.body)["status"] == "completed"
    assert delete.status_code == 200
    assert json.loads(delete.body)["status"] == "deleted"
    assert create_route.call_count == 1
    assert json.loads(create_route.calls[0].request.content) == create_body
    assert delete_route.call_count == 1
    client_poll: Final = poll_route.calls[-1].request
    assert client_poll.headers["authorization"] == _KEY
    assert create_route.calls[0].request.headers["authorization"] == _KEY
    assert delete_route.calls[0].request.headers["authorization"] == _KEY


@pytest.mark.asyncio
async def test_assemblyai_transcribe_with_non_admin_key_logs_key_identity(
    respx_mock: respx.MockRouter, httpx_transport: None, recorder: _SuccessRecorder
) -> None:
    create_route: Final = respx_mock.post(_UPSTREAM).mock(
        return_value=httpx.Response(200, json={"id": "tr_9", "status": "queued", "audio_url": _AUDIO_URL})
    )
    respx_mock.get(f"{_UPSTREAM}/tr_9").mock(
        return_value=httpx.Response(200, json={"id": "tr_9", "status": "completed", "text": "fires near town"})
    )
    non_admin: Final = UserAPIKeyAuth(
        api_key="hashed-non-admin-key",
        token="hashed-non-admin-key",
        user_id="non-admin-user",
        team_id="non-admin-team",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    response: Final = await assemblyai_proxy_route(
        endpoint="v2/transcript",
        request=_request("POST", "/assemblyai/v2/transcript", json.dumps({"audio_url": _AUDIO_URL}).encode()),
        fastapi_response=Response(),
        user_api_key_dict=non_admin,
    )
    logged: Final = await _wait_for_success_log(recorder, "tr_9")

    assert isinstance(response, Response)
    assert response.status_code == 200, response.body
    assert create_route.call_count == 1
    assert create_route.calls[0].request.headers["authorization"] == _KEY
    assert logged["metadata"]["user_api_key_hash"] == "hashed-non-admin-key"
    assert logged["metadata"]["user_api_key_user_id"] == "non-admin-user"
    assert logged["metadata"]["user_api_key_team_id"] == "non-admin-team"
    assert logged["custom_llm_provider"] == "assemblyai"
