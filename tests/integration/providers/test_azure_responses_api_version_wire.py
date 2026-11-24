from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
from integration._support.client import Gateway
from integration._support.openai_wire import responses_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_DEPLOYMENT: Final = "gpt-5.4-nano"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _reply(_: Request) -> Reply:
    return responses_reply("resp_azure_version", _DEPLOYMENT, "azure answered", stream=False)


def _only_post(wire: Wire) -> tuple[str, dict[str, str], dict[str, JsonValue]]:
    posts: Final = tuple(request for request in wire.drain() if request.method == "POST")
    assert len(posts) == 1, [request.target for request in posts]
    target: Final = urlsplit(posts[0].target)
    query: Final = {name: values[-1] for name, values in parse_qs(target.query).items()}
    return target.path, query, _JSON_OBJECT.validate_json(posts[0].body)


@pytest.mark.parametrize("api_version", [None, "v1", "2025-04-01-preview"])
def test_responses_on_a_foundry_project_v1_base_sends_no_api_version(gateway: Gateway, api_version: str | None) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        project_base: Final = f"{wire.url}/api/projects/integration-project/openai/v1"
        model: Final = (
            scenario.model(model=f"azure/{_DEPLOYMENT}", api_base=project_base, api_key="synthetic-azure-key")
            if api_version is None
            else scenario.model(
                model=f"azure/{_DEPLOYMENT}",
                api_base=project_base,
                api_key="synthetic-azure-key",
                api_version=api_version,
            )
        )
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": "project responses"}, key=scenario.key()
        )

        assert response.status_code == 200, response.text
        path, query, body = _only_post(wire)
        assert (path, query) == ("/api/projects/integration-project/openai/v1/responses", {})
        assert body["input"] == "project responses", body
