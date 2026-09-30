from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from uuid import uuid4

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse


@dataclass(frozen=True, slots=True)
class MissingBodyParamCase:
    case_id: str
    path: str
    param: str
    fields: Mapping[str, JsonValue]
    provider_model: str
    multipart: bool = False
    search_tool: bool = False


MISSING_BODY_PARAM_CASES: Final = (
    MissingBodyParamCase("speech_input", "/v1/audio/speech", "input", {"voice": "alloy"}, "openai/tts-1"),
    MissingBodyParamCase("moderations_input", "/v1/moderations", "input", {}, "openai/omni-moderation-latest"),
    MissingBodyParamCase("image_generation_prompt", "/v1/images/generations", "prompt", {}, "openai/gpt-image-1"),
    MissingBodyParamCase("completion_prompt", "/v1/completions", "prompt", {}, "openai/gpt-3.5-turbo-instruct"),
    MissingBodyParamCase("rerank_query", "/v1/rerank", "query", {"documents": ["document"]}, "cohere/rerank-v3.5"),
    MissingBodyParamCase("rerank_documents", "/v1/rerank", "documents", {"query": "query"}, "cohere/rerank-v3.5"),
    MissingBodyParamCase(
        "image_edit_image", "/v1/images/edits", "image", {"prompt": "edit this image"}, "openai/gpt-image-1", True
    ),
    MissingBodyParamCase("search_query", "/v1/search", "query", {}, "openai/gpt-4o-mini", search_tool=True),
)


def _scripted_api_base(scenario: Scenario) -> str:
    handle = register_scenario(
        f"error-status-{uuid4().hex}",
        JsonResponse(
            content_type="application/json",
            body={"error": {"message": "Unexpected provider dispatch", "type": "invalid_request_error"}},
            status=400,
        ),
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return f"{handle.api_base()}/v1"


def _delete_search_tool(gateway: Gateway, search_tool_id: str) -> None:
    response = gateway.request("DELETE", f"/search_tools/{search_tool_id}")
    assert response.status_code == 200, response.text


def _register_search_tool(gateway: Gateway, scenario: Scenario, api_base: str) -> str:
    name = f"error-status-search-{uuid4().hex}"
    created = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {
                    "search_provider": "perplexity",
                    "api_key": "integration-provider-key",
                    "api_base": api_base,
                },
            }
        },
    )
    scenario.cleanups.callback(_delete_search_tool, gateway, string_value(created["search_tool_id"]))
    return name


def _read_observations(upstream: httpx.Client) -> tuple[JsonValue, ...]:
    response = upstream.get("/__observations")
    assert response.status_code == 200, response.text
    body = JSON_OBJECT.validate_json(response.content)
    requests = body.get("requests")
    assert isinstance(requests, list), response.text
    return tuple(requests)


@pytest.mark.parametrize(
    "case",
    MISSING_BODY_PARAM_CASES,
    ids=(
        "speech_input",
        "moderations_input",
        "image_generation_prompt",
        "completion_prompt",
        "rerank_query",
        "rerank_documents",
        "image_edit_image",
        "search_query",
    ),
)
def test_missing_required_body_param_is_400_naming_the_param(gateway: Gateway, case: MissingBodyParamCase) -> None:
    with gateway.scenario() as scenario:
        api_base: Final = _scripted_api_base(scenario)
        model: Final = scenario.model(model=case.provider_model, api_base=api_base)
        key: Final = scenario.key() if case.search_tool else scenario.key(models=[model])
        search_tool_name: Final = _register_search_tool(gateway, scenario, api_base) if case.search_tool else None
        body: Final = (
            {"search_tool_name": search_tool_name} if search_tool_name is not None else {"model": model, **case.fields}
        )
        with httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
            _read_observations(upstream)
            response: Final = (
                gateway.client.post(
                    case.path,
                    data={"model": model},
                    files=[("prompt", (None, string_value(body["prompt"])))],
                    headers={"Authorization": f"Bearer {key}"},
                )
                if case.multipart
                else gateway.request("POST", case.path, body, key=key)
            )
            observations: Final = _read_observations(upstream)
        response_body: Final = JSON_OBJECT.validate_json(response.content)
        error_value: Final = response_body.get("error")
        assert isinstance(error_value, dict), response.text
        error: Final = object_value(error_value)
        assert observations == (), f"Unexpected scripted upstream requests: {observations}\n{response.text}"
        assert (response.status_code, error.get("type"), error.get("param")) == (
            400,
            "invalid_request_error",
            case.param,
        ), response.text


@pytest.mark.parametrize(
    ("fields", "missing"),
    (
        ({"max_tokens": 100}, "messages"),
        ({"messages": [{"role": "user", "content": "missing max tokens"}]}, "max_tokens"),
    ),
    ids=("messages", "max_tokens"),
)
def test_anthropic_messages_missing_required_param_is_400(
    gateway: Gateway, fields: Mapping[str, JsonValue], missing: str
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_base=_scripted_api_base(scenario),
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, **fields},
            key=key,
        )
        response_body: Final = JSON_OBJECT.validate_json(response.content)
        error_value: Final = response_body.get("error")
        assert isinstance(error_value, dict), response.text
        error: Final = object_value(error_value)
        assert (response.status_code, response_body.get("type"), error.get("type")) == (
            400,
            "error",
            "invalid_request_error",
        ), f"Missing {missing}: {response.text}"


def test_unknown_search_tool_is_400(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        api_base: Final = _scripted_api_base(scenario)
        _register_search_tool(gateway, scenario, api_base)
        key: Final = scenario.key()
        response: Final = gateway.request(
            "POST",
            "/v1/search",
            {"search_tool_name": f"unconfigured-{uuid4().hex}", "query": "query"},
            key=key,
        )
        response_body: Final = JSON_OBJECT.validate_json(response.content)
        error_value: Final = response_body.get("error")
        assert isinstance(error_value, dict), response.text
        error: Final = object_value(error_value)
        assert (response.status_code, error.get("type")) == (
            400,
            "invalid_request_error",
        ), response.text


@pytest.mark.parametrize("method", ("GET", "DELETE"), ids=("get", "delete"))
def test_unknown_response_id_is_404(gateway: Gateway, method: str) -> None:
    response_id: Final = "resp_0123456789abcdef0123456789abcdef0123456789abcdef"
    response: Final = gateway.request("GET" if method == "GET" else "DELETE", f"/v1/responses/{response_id}")
    assert response.status_code == 404, response.text


def test_invalid_pagination_is_400(gateway: Gateway) -> None:
    response: Final = gateway.request(
        "GET",
        "/team/daily/activity?start_date=2024-01-01&end_date=2024-01-02&page=0",
    )
    assert response.status_code == 400, response.text


def test_invalid_spend_logs_date_is_400(gateway: Gateway) -> None:
    response: Final = gateway.request(
        "GET",
        "/spend/logs?start_date=notadate&end_date=alsonot",
    )
    assert response.status_code == 400, response.text
