import json
import urllib.parse
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

_ENV_GEMINI_KEY: Final = "synthetic-shared-env-gemini-key"
_AGENT_NAME: Final = "integration-slides-agent"

_AGENT_REPLY: Final = {"id": _AGENT_NAME, "base_agent": "waverunner"}


def _template(params: dict) -> str:
    return urllib.parse.quote(json.dumps(params))


@pytest.mark.parametrize("caller", ["admin", "virtual_key"], ids=["admin", "virtual_key"])
@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/v1beta/agents"),
        ("GET", "/v1beta/agents"),
        ("GET", f"/v1beta/agents/{_AGENT_NAME}"),
        ("GET", f"/v1beta/agents/{_AGENT_NAME}/versions"),
        ("DELETE", f"/v1beta/agents/{_AGENT_NAME}"),
    ],
)
def test_agents_rejects_api_base_without_caller_key(
    gateway: Gateway, tmp_path: Path, method: str, path: str, caller: str
) -> None:
    def respond(request: Request) -> Reply:
        return Reply(body=b"{}")

    with wire_server(respond) as wire:
        with owned_proxy_process(
            gateway, tmp_path, {"GEMINI_API_KEY": _ENV_GEMINI_KEY}
        ) as owned, owned.gateway.scenario() as scenario:
            virtual_key: Final = None if caller == "admin" else scenario.key()
            if method == "POST":
                response: Final = owned.gateway.request(
                    "POST",
                    path,
                    {
                        "name": _AGENT_NAME,
                        "base_agent": "waverunner",
                        "instructions": "make slides",
                        "litellm_params_template": {"api_base": wire.url},
                    },
                    key=virtual_key,
                )
            else:
                response = owned.gateway.request(
                    method,
                    f"{path}?litellm_params_template={_template({'api_base': wire.url})}",
                    key=virtual_key,
                )
            if caller == "admin":
                assert response.status_code == 500, response.text
                assert response.json()["error"]["type"] == "internal_server_error", response.text
                assert "api_base" in response.text and "api_key" in response.text, response.text
            else:
                assert response.status_code == 401, response.text
                assert "caller-supplied" in response.json()["detail"], response.text
            assert wire.drain() == (), response.text


def test_agents_forwards_caller_supplied_key_to_chosen_api_base(gateway: Gateway, tmp_path: Path) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1beta/agents"
        assert request.headers["x-goog-api-key"] == "synthetic-caller-key"
        assert _ENV_GEMINI_KEY not in request.headers.values()
        assert json.loads(request.body) == {
            "name": _AGENT_NAME,
            "base_agent": "waverunner",
            "instructions": "make slides",
        }
        return Reply(body=json.dumps(_AGENT_REPLY).encode())

    with wire_server(respond) as wire:
        with owned_proxy_process(gateway, tmp_path, {"GEMINI_API_KEY": _ENV_GEMINI_KEY}) as owned:
            response: Final = owned.gateway.request(
                "POST",
                "/v1beta/agents",
                {
                    "name": _AGENT_NAME,
                    "base_agent": "waverunner",
                    "instructions": "make slides",
                    "litellm_params_template": {"api_base": wire.url, "api_key": "synthetic-caller-key"},
                },
            )
            assert response.status_code == 200, response.text
            assert json.loads(response.content) == {**_AGENT_REPLY, "name": _AGENT_NAME}, response.text
            assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1beta/agents")]
