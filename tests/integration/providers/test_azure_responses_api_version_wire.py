from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_DEPLOYMENT: Final = "gpt-5.4-nano"
_DATED_GA_API_VERSION: Final = "2024-10-21"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _reply(request: Request) -> Reply:
    if urlsplit(request.target).path.endswith("/responses"):
        return responses_reply("resp_azure_version", _DEPLOYMENT, "azure answered", stream=False)
    return chat_reply("chatcmpl-azure-version", _DEPLOYMENT, "azure answered", stream=False)


def _dated_azure_model(scenario: Scenario, wire: Wire, model: str, responses_api_version: str | None = None) -> str:
    if responses_api_version is None:
        return scenario.model(
            model=model, api_base=wire.url, api_key="synthetic-azure-key", api_version=_DATED_GA_API_VERSION
        )
    return scenario.model(
        model=model,
        api_base=wire.url,
        api_key="synthetic-azure-key",
        api_version=_DATED_GA_API_VERSION,
        responses_api_version=responses_api_version,
    )


def _only_post(wire: Wire) -> tuple[str, dict[str, str], dict[str, JsonValue]]:
    posts: Final = tuple(request for request in wire.drain() if request.method == "POST")
    assert len(posts) == 1, [request.target for request in posts]
    target: Final = urlsplit(posts[0].target)
    query: Final = {name: values[-1] for name, values in parse_qs(target.query).items()}
    return target.path, query, _JSON_OBJECT.validate_json(posts[0].body)


def test_chat_keeps_the_dated_api_version_and_never_sends_responses_api_version(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = _dated_azure_model(scenario, wire, f"azure/{_DEPLOYMENT}", responses_api_version="preview")
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "dated chat"}]},
            key=scenario.key(),
        )

        assert response.status_code == 200, response.text
        path, query, body = _only_post(wire)
        assert (path, query) == (
            f"/openai/deployments/{_DEPLOYMENT}/chat/completions",
            {"api-version": _DATED_GA_API_VERSION},
        )
        assert body == {
            "model": _DEPLOYMENT,
            "messages": [{"role": "user", "content": "dated chat"}],
            "stream": False,
        }, body


def test_responses_on_a_dated_ga_deployment_reaches_the_v1_responses_route(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = _dated_azure_model(scenario, wire, f"azure/{_DEPLOYMENT}")
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": "dated responses"}, key=scenario.key()
        )

        assert response.status_code == 200, response.text
        path, query, body = _only_post(wire)
        assert (path, query) == ("/openai/v1/responses", {"api-version": "preview"})
        assert body["input"] == "dated responses", body


def test_responses_api_version_overrides_the_chat_api_version_and_stays_out_of_the_body(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = _dated_azure_model(
            scenario, wire, f"azure/{_DEPLOYMENT}", responses_api_version="2025-04-01-preview"
        )
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": "override responses"}, key=scenario.key()
        )

        assert response.status_code == 200, response.text
        path, query, body = _only_post(wire)
        assert (path, query) == ("/openai/responses", {"api-version": "2025-04-01-preview"})
        assert "responses_api_version" not in body and "api_version" not in body, body


def test_chat_bridged_through_azure_responses_on_a_dated_ga_deployment_reaches_v1_responses(
    gateway: Gateway,
) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = _dated_azure_model(scenario, wire, f"azure/responses/{_DEPLOYMENT}")
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "bridged chat"}]},
            key=scenario.key(),
        )

        assert response.status_code == 200, response.text
        assert "azure answered" in response.text, response.text
        path, query, _ = _only_post(wire)
        assert (path, query) == ("/openai/v1/responses", {"api-version": "preview"})


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
