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


@pytest.fixture(scope="module")
def connected_copilot(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
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
            assert request.headers.get("authorization") == f"Bearer {_SESSION_TOKEN}"
            if request.method == "POST":
                assert request.target == "/responses"
                body: Final = JSON_OBJECT.validate_json(request.body)
                assert body["model"] == _MODEL
                assert body["input"] in (
                    _PROMPT,
                    [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": _PROMPT}]}],
                )
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
                owned_proxy_process(gateway, directory, peers.environment(), state=OwnedState.PRIVATE) as owned,
                owned.gateway.scenario() as scenario,
            ):
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
                yield caller_proxy, key
        finally:
            delete_scenario(script)


@pytest.mark.parametrize("route", ["/v1/responses", "/v1/messages", "/v1/chat/completions"])
def test_copilot_responses_model_uses_the_connected_key_owner(
    route: str, connected_copilot: tuple[Gateway, str]
) -> None:
    gateway, key = connected_copilot
    messages: Final[list[JsonValue]] = [{"role": "user", "content": _PROMPT}]
    payloads: Final[dict[str, dict[str, JsonValue]]] = {
        "/v1/responses": {"input": _PROMPT},
        "/v1/messages": {"max_tokens": 64, "messages": messages},
        "/v1/chat/completions": {"max_tokens": 64, "messages": messages},
    }
    response: Final = gateway.request(
        "POST", route, {"model": _MODEL, **payloads[route]}, key=key, headers={"anthropic-version": "2023-06-01"}
    )
    assert response.status_code == 200, response.text
    body: Final = response.json()
    if route == "/v1/messages":
        assert body["content"][0]["text"] == _ANSWER
    elif route == "/v1/chat/completions":
        assert body["choices"][0]["message"]["content"] == _ANSWER
    else:
        assert body["output"][0]["content"][0]["text"] == _ANSWER
