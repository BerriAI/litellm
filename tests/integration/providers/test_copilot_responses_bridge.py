import json
import os
import subprocess
import sys
from collections.abc import Iterator
from typing import Final
from uuid import uuid4

import httpx
import pytest
from integration._support import device_login
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, string_value
from integration._support.process import OwnedState, owned_proxy_process
from integration._support.upstream import JsonResponse, RoutedResponse, delete_scenario, register_scenario
from integration._support.wire import Reply, Request
from pydantic import JsonValue

_MODEL: Final = "gpt-6.1-sol"
_PROMPT: Final = "Hello Copilot"
_ANSWER: Final = "Hello from the connected Copilot account"
_SESSION_TOKEN: Final = device_login.COPILOT_FIRST_LOGIN_ACCESS
_SHARED_TOKEN: Final = "scripted-copilot-shared-token"


def _response() -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$UNIQUE_ID",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": _MODEL,
            "output": [
                {
                    "id": "msg_$UNIQUE_ID",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": _ANSWER, "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10},
        },
    )


def _stream_response() -> Reply:
    events: Final = (
        {
            "type": "response.output_text.delta",
            "item_id": "msg_stream",
            "output_index": 0,
            "content_index": 0,
            "delta": _ANSWER,
        },
        {"type": "response.completed", "response": _response().body},
    )
    return Reply(200, "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode(), "text/event-stream")


@pytest.fixture(scope="module")
def connected_copilot(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Gateway, str, str, device_login.Peers, str]]:
    directory: Final = tmp_path_factory.mktemp("copilot-responses-bridge")
    credential: Final = f"copilot-bridge-{uuid4().hex}"
    with gateway_from_environment() as gateway, httpx.Client(trust_env=False, timeout=10) as upstream:
        script: Final = register_scenario(
            credential,
            RoutedResponse(
                content_type="application/x-routed",
                routes={
                    "GET /models": JsonResponse(content_type="application/json", body={"data": [{"id": _MODEL}]}),
                    "POST /responses": _response(),
                },
            ),
            control_url=gateway.upstream_url,
        )

        def respond(request: Request) -> Reply:
            assert request.headers.get("authorization") in (f"Bearer {_SESSION_TOKEN}", f"Bearer {_SHARED_TOKEN}")
            if request.method == "POST":
                assert request.target == "/responses"
                body: Final = JSON_OBJECT.validate_json(request.body)
                assert body["model"] == _MODEL
                assert body["input"] in (
                    _PROMPT,
                    [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": _PROMPT}]}],
                )
                if body.get("stream") is True:
                    return _stream_response()
            response: Final = upstream.request(
                request.method,
                f"{script.api_base()}{request.target}",
                content=request.body,
                headers={
                    "authorization": request.headers["authorization"],
                    "content-type": request.headers.get("content-type", "application/json"),
                },
            )
            return Reply(response.status_code, response.content, response.headers["content-type"])

        try:
            with (
                device_login.device_login_peers(directory, respond) as peers,
                owned_proxy_process(
                    gateway,
                    directory,
                    {**peers.environment(), "GITHUB_COPILOT_TOKEN_DIR": str(directory)},
                    workers=2,
                    state=OwnedState.PRIVATE,
                ) as owned,
                owned.gateway.scenario() as scenario,
            ):
                device_login.write_copilot_access(directory, _SHARED_TOKEN)
                caller_proxy: Final = owned.gateway
                caller_proxy.post(
                    "/credentials",
                    {
                        "credential_name": credential,
                        "credential_values": {"github_copilot_auth_type": "per_user_oauth"},
                        "credential_info": {"custom_llm_provider": "github_copilot"},
                    },
                )
                model: Final = scenario.model(
                    name=_MODEL,
                    model=f"github_copilot/{_MODEL}",
                    api_key=None,
                    api_base=None,
                    litellm_credential_name=credential,
                )
                user: Final = scenario.user(user_role="internal_user")
                key: Final = scenario.key(user_id=user, models=[model])
                peers.switches.grant.set()
                path: Final = f"/credentials/{credential}/user_connection"
                started: Final = caller_proxy.request("POST", f"{path}/start", {}, key=key)
                assert started.status_code == 200, started.text
                finished: Final = caller_proxy.request(
                    "POST", f"{path}/poll", {"flow_handle": string_value(started.json()["flow_handle"])}, key=key
                )
                assert finished.status_code == 200, finished.text
                assert finished.json()["status"] == "connected", finished.text
                shared_credential: Final = f"{credential}-shared"
                caller_proxy.post(
                    "/credentials",
                    {
                        "credential_name": shared_credential,
                        "credential_values": {"github_copilot_host": "github.com"},
                        "credential_info": {"custom_llm_provider": "github_copilot"},
                    },
                )
                shared_model: Final = scenario.model(
                    name=f"{_MODEL}-shared",
                    model=f"github_copilot/{_MODEL}",
                    api_key=None,
                    api_base=None,
                    litellm_credential_name=shared_credential,
                )
                disconnected_user: Final = scenario.user(user_role="internal_user")
                disconnected_key: Final = scenario.key(user_id=disconnected_user, models=[model, shared_model])
                yield caller_proxy, key, disconnected_key, peers, shared_model
        finally:
            delete_scenario(script)


@pytest.mark.parametrize(
    ("route", "stream"),
    [
        ("/v1/responses", False),
        ("/v1/messages", False),
        ("/v1/messages", True),
        ("/v1/chat/completions", False),
        ("/v1/chat/completions", True),
    ],
    ids=["responses-control", "messages-buffered", "messages-streaming", "chat-buffered", "chat-streaming"],
)
def test_copilot_responses_model_uses_the_connected_key_owner(
    route: str,
    stream: bool,
    connected_copilot: tuple[Gateway, str, str, device_login.Peers, str],
) -> None:
    gateway, key, _, peers, _ = connected_copilot
    peers.api.drain()
    messages: Final[list[JsonValue]] = [{"role": "user", "content": _PROMPT}]
    payloads: Final[dict[str, dict[str, JsonValue]]] = {
        "/v1/responses": {"input": _PROMPT},
        "/v1/messages": {"max_tokens": 64, "messages": messages},
        "/v1/chat/completions": {"max_tokens": 64, "messages": messages},
    }
    response: Final = gateway.request(
        "POST",
        route,
        {"model": _MODEL, "stream": stream, **payloads[route]},
        key=key,
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text
    if stream:
        assert _ANSWER in response.text
        assert ("message_stop" if route == "/v1/messages" else "data: [DONE]") in response.text
    else:
        body: Final = response.json()
        if route == "/v1/messages":
            assert body["content"][0]["text"] == _ANSWER
        elif route == "/v1/chat/completions":
            assert body["choices"][0]["message"]["content"] == _ANSWER
        else:
            assert body["output"][0]["content"][0]["text"] == _ANSWER
    sent: Final = tuple(request for request in peers.api.drain() if request.method == "POST")
    assert sent and all(request.headers.get("authorization") == f"Bearer {_SESSION_TOKEN}" for request in sent)


def test_copilot_bridge_rejects_disconnected_key_without_upstream_call(
    connected_copilot: tuple[Gateway, str, str, device_login.Peers, str],
) -> None:
    gateway, _, key, peers, _ = connected_copilot
    peers.api.drain()
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": _MODEL, "messages": [{"role": "user", "content": _PROMPT}]},
        key=key,
    )
    assert response.status_code == 401, response.text
    assert "has not connected GitHub Copilot" in response.json()["error"]["message"]
    assert not peers.api.drain()


def test_copilot_bridge_keeps_shared_named_credential_behavior(
    connected_copilot: tuple[Gateway, str, str, device_login.Peers, str],
) -> None:
    gateway, _, key, peers, shared_model = connected_copilot
    peers.api.drain()
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": shared_model, "messages": [{"role": "user", "content": _PROMPT}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == _ANSWER
    sent: Final = tuple(request for request in peers.api.drain() if request.method == "POST")
    assert sent and all(request.headers.get("authorization") == f"Bearer {_SHARED_TOKEN}" for request in sent)


_SDK_PROBE: Final = """
import asyncio
import json
import os
import litellm

args = json.loads(os.environ["COPILOT_AUDIT_ARGS"])

async def main():
    try:
        if args["async"]:
            result = await litellm.acompletion(**args["request"])
            if args["request"]["stream"]:
                text = "".join([chunk.choices[0].delta.content or "" async for chunk in result if chunk.choices])
        else:
            result = litellm.completion(**args["request"])
            if args["request"]["stream"]:
                text = "".join(chunk.choices[0].delta.content or "" for chunk in result if chunk.choices)
        if not args["request"]["stream"]:
            text = result.choices[0].message.content
        print(json.dumps({"text": text}))
    except litellm.BadRequestError as error:
        print(json.dumps({"status": error.status_code, "message": str(error)}))

asyncio.run(main())
"""


@pytest.mark.parametrize(
    ("is_async", "stream", "has_credential"),
    [(False, False, True), (True, False, True), (False, True, True), (True, True, True), (False, False, False)],
    ids=["sync-buffered", "async-buffered", "sync-streaming", "async-streaming", "missing-credential"],
)
def test_copilot_bridge_sdk_against_scripted_peer(
    is_async: bool,
    stream: bool,
    has_credential: bool,
    connected_copilot: tuple[Gateway, str, str, device_login.Peers, str],
) -> None:
    _, _, _, peers, _ = connected_copilot
    peers.api.drain()
    request: Final[dict[str, JsonValue]] = {
        "model": f"github_copilot/{_MODEL}",
        "messages": [{"role": "user", "content": _PROMPT}],
        "stream": stream,
        "github_copilot_auth_type": "per_user_oauth",
        **(
            {
                "litellm_credential_name": "sdk-copilot",
                "secret_fields": {
                    "user_provider_credentials_user_id": "sdk-user",
                    "user_provider_credentials": {
                        "sdk-copilot": {"access_token": _SESSION_TOKEN, "github_host": "github.com"}
                    },
                },
            }
            if has_credential
            else {}
        ),
    }
    completed: Final = subprocess.run(
        [sys.executable, "-P", "-c", _SDK_PROBE],
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
        env={
            **os.environ,
            **peers.environment(),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "COPILOT_AUDIT_ARGS": json.dumps({"async": is_async, "request": request}),
        },
    )
    result: Final = JSON_OBJECT.validate_json(completed.stdout.strip().splitlines()[-1])
    if has_credential:
        assert result == {"text": _ANSWER}
        sent: Final = tuple(request for request in peers.api.drain() if request.method == "POST")
        assert sent and all(request.headers.get("authorization") == f"Bearer {_SESSION_TOKEN}" for request in sent)
    else:
        assert result["status"] == 400
        assert "requires litellm_credential_name" in string_value(result["message"])
        assert not peers.api.drain()
