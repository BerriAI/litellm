import base64
import json
import uuid
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import pytest
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "databricks/synthetic-vendor.chat-model.v1"
_CLIENT_ID: Final = "synthetic-databricks-client-id"
_CLIENT_SECRET: Final = "synthetic-databricks-client-secret"
_ACCESS_TOKEN: Final = "synthetic-databricks-oauth-token"
_PROMPT: Final = "Which workspace issued this token?"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _basic_credentials(client_id: str, client_secret: str) -> str:
    return "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()


def _completion(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _MODEL.removeprefix("databricks/"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "the workspace origin"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        }
    ).encode()


@pytest.mark.covers("other.provider_wire.databricks.oauth_token_url_uses_workspace_origin_for_ai_gateway_api_base")
def test_databricks_ai_gateway_api_base_requests_oauth_token_from_workspace_origin(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"databricks-oauth-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.target == "/oidc/v1/token":
            assert request.method == "POST"
            assert request.headers["authorization"] == _basic_credentials(_CLIENT_ID, _CLIENT_SECRET)
            assert request.headers["content-type"] == "application/x-www-form-urlencoded"
            assert parse_qs(request.body.decode()) == {"grant_type": ["client_credentials"], "scope": ["all-apis"]}
            return Reply(
                body=json.dumps({"access_token": _ACCESS_TOKEN, "token_type": "Bearer", "expires_in": 3600}).encode()
            )
        if request.target == "/ai-gateway/mlflow/v1/chat/completions":
            assert request.method == "POST"
            assert request.headers["authorization"] == f"Bearer {_ACCESS_TOKEN}"
            body: Final = _JSON_OBJECT.validate_json(request.body)
            assert body["model"] == _MODEL.removeprefix("databricks/")
            assert body["messages"] == [{"role": "user", "content": _PROMPT}]
            return Reply(body=_completion(identity))
        return Reply(status=401, body=json.dumps({"error": f"unauthenticated path {request.target}"}).encode())

    overrides: Final = {"DATABRICKS_CLIENT_ID": _CLIENT_ID, "DATABRICKS_CLIENT_SECRET": _CLIENT_SECRET}
    with wire_server(respond) as wire, owned_proxy(gateway, tmp_path, overrides) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/ai-gateway/mlflow/v1", api_key=None)
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": _PROMPT}]},
            )
            assert response.status_code == 200, response.text
            payload: Final = _JSON_OBJECT.validate_json(response.content)
            assert payload["id"] == identity
            assert payload["choices"] == [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "the workspace origin", "role": "assistant"},
                }
            ]
            assert payload["usage"] == {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
            assert [(request.method, request.target) for request in wire.drain()] == [
                ("POST", "/oidc/v1/token"),
                ("POST", "/ai-gateway/mlflow/v1/chat/completions"),
            ]
