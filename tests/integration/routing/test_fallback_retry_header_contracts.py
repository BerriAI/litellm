import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Literal

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.database import read_rows
from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    object_value,
    string_value,
)
from tests.integration._support.openai_wire import chat_reply
from tests.integration._support.wire import Reply, Request, wire_server

UNREACHABLE_API_BASE: Final = "http://127.0.0.1:9/v1"
END_USER_REQUESTS: Final = 10
CONCURRENT_REQUESTS: Final = 25
DISTRIBUTION_REQUESTS: Final = 20
SLOW_UPSTREAM_SECONDS: Final = 3
HELD_REPLY_SECONDS: Final = 30
END_USER_ROW_SECONDS: Final = 70
CUSTOM_FALLBACK_TEXT: Final = "custom fallback prompt"
Caller = Literal["virtual-key", "master-key"]
JSON_OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _messages(text: str = "integration control") -> list[JsonValue]:
    return [{"role": "user", "content": text}]


def _chat(
    gateway: Gateway,
    body: dict[str, JsonValue],
    *,
    key: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", {"messages": _messages(), **body}, key=key, headers=headers)


def _body(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def _content(response: httpx.Response) -> str:
    choices: Final = _body(response)["choices"]
    assert isinstance(choices, list) and choices, response.text
    return string_value(object_value(object_value(choices[0])["message"])["content"])


def _scripted_model(gateway: Gateway, scenario: Scenario, statuses: list[int], num_retries: int) -> str:
    upstream_model: Final = f"integration-{uuid.uuid4().hex}"
    script_url: Final = f"{gateway.upstream_url}/__scripts/{upstream_model}"
    configured: Final = httpx.post(script_url, json={"statuses": statuses})
    assert configured.status_code == 200, configured.text
    scenario.cleanups.callback(httpx.delete, script_url)
    return scenario.model(model=f"openai/{upstream_model}", num_retries=num_retries)


@dataclass(frozen=True, slots=True)
class UniqueModel:
    name: str
    upstream: str
    deployment_id: str


def _unique_model(scenario: Scenario) -> UniqueModel:
    upstream_model: Final = f"integration-{uuid.uuid4().hex}"
    deployment_id: Final = f"integration-{uuid.uuid4().hex}"
    model_info: Final[dict[str, JsonValue]] = {"id": deployment_id}
    name: Final = scenario.model(model=f"openai/{upstream_model}", model_info=model_info)
    return UniqueModel(name, upstream_model, deployment_id)


def _slow_reply(_: Request) -> Reply:
    time.sleep(SLOW_UPSTREAM_SECONDS)
    return chat_reply("chatcmpl-slow", "gpt-4o-mini", "late", stream=False)


def _fallback_reply(_: Request) -> Reply:
    return chat_reply("chatcmpl-fallback", "gpt-4o-mini", "served by fallback", stream=False)


def _held_reply(release: threading.Event) -> Callable[[Request], Reply]:
    def respond(_: Request) -> Reply:
        assert release.wait(timeout=HELD_REPLY_SECONDS)
        return chat_reply("chatcmpl-held", "gpt-4o-mini", "held", stream=False)

    return respond


def _delete_auto_created_end_user(gateway: Gateway, user_id: str) -> None:
    eventually(
        lambda: read_rows('SELECT user_id FROM "LiteLLM_EndUserTable" WHERE user_id = %s', (user_id,)),
        lambda rows: len(rows) == 1,
        seconds=END_USER_ROW_SECONDS,
    )
    gateway.post("/end_user/delete", {"user_ids": [user_id]})
    assert read_rows('SELECT user_id FROM "LiteLLM_EndUserTable" WHERE user_id = %s', (user_id,)) == []


def _status(gateway: Gateway, model: str) -> int:
    return _chat(gateway, {"model": model}).status_code


def _served_model_id(gateway: Gateway, model: str) -> str:
    response: Final = _chat(gateway, {"model": model})
    assert response.status_code == 200, response.text
    return response.headers["x-litellm-model-id"]


def _concurrent_round(gateway: Gateway, bad: str, good: str) -> tuple[list[int], list[int]]:
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS * 2) as pool:
        bad_calls: Final = [pool.submit(_status, gateway, bad) for _ in range(CONCURRENT_REQUESTS)]
        good_calls: Final = [pool.submit(_status, gateway, good) for _ in range(CONCURRENT_REQUESTS)]
        return [call.result() for call in bad_calls], [call.result() for call in good_calls]


def _deployment(gateway: Gateway, scenario: Scenario, model_name: str, model: str = "openai/gpt-4o-mini") -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": model_name,
            "litellm_params": {
                "model": model,
                "api_key": "integration-provider-key",
                "api_base": f"{gateway.upstream_url}/v1",
            },
        },
    )
    deployment_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, deployment_id)
    return deployment_id


@pytest.mark.parametrize("caller", ["virtual-key", "master-key"])
def test_end_user_budget_tpm_limit_rate_limits_their_requests(gateway: Gateway, caller: Caller) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(tpm_limit=2)
        end_user: Final = f"integration-{uuid.uuid4().hex}"
        gateway.post("/end_user/new", {"user_id": end_user, "budget_id": budget})
        scenario.cleanups.callback(gateway.post, "/end_user/delete", {"user_ids": [end_user]})
        key: Final = scenario.key(models=[model]) if caller == "virtual-key" else gateway.key
        control_user: Final = f"integration-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_auto_created_end_user, gateway, control_user)
        control: Final = _chat(gateway, {"model": model, "user": control_user}, key=key)
        assert control.status_code == 200, control.text
        statuses: Final = [
            _chat(gateway, {"model": model, "user": end_user}, key=key).status_code for _ in range(END_USER_REQUESTS)
        ]
        assert statuses.count(200) < 5, statuses
        assert set(statuses) <= {200, 429}, statuses


def test_client_fallbacks_reach_an_allowed_model_and_name_a_denied_one(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        primary: Final = scenario.model(api_base=UNREACHABLE_API_BASE)
        fallback: Final = _unique_model(scenario)
        body: Final[dict[str, JsonValue]] = {"model": primary, "fallbacks": [fallback.name]}
        served: Final = _chat(gateway, body, key=scenario.key(models=[primary, fallback.name]))
        assert served.status_code == 200, served.text
        assert _body(served)["model"] == fallback.upstream
        assert served.headers["x-litellm-model-id"] == fallback.deployment_id
        assert _content(served)
        denied: Final = _chat(gateway, body, key=scenario.key(models=[primary]))
        assert denied.status_code == 403, denied.text
        assert fallback.name in denied.text


def test_client_fallback_with_custom_messages_sends_them_to_the_fallback(gateway: Gateway) -> None:
    custom: Final = _messages(CUSTOM_FALLBACK_TEXT)
    with gateway.scenario() as scenario, wire_server(_fallback_reply) as wire:
        primary: Final = scenario.model(api_base=UNREACHABLE_API_BASE)
        fallback: Final = scenario.model(api_base=wire.url)
        body: Final[dict[str, JsonValue]] = {
            "model": primary,
            "fallbacks": [{"model": fallback, "messages": custom}],
        }
        served: Final = _chat(gateway, body, key=scenario.key(models=[primary, fallback]))
        assert served.status_code == 200, served.text
        assert _content(served) == "served by fallback"
        forwarded: Final = [JSON_OBJECT.validate_json(request.body)["messages"] for request in wire.drain()]
        assert forwarded == [custom]
        denied: Final = _chat(gateway, body, key=scenario.key(models=[primary]))
        assert denied.status_code == 403, denied.text
        assert fallback in denied.text
        assert wire.drain() == ()


def test_rate_limited_deployment_is_retried_and_reports_retry_counts(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _scripted_model(gateway, scenario, [429, 200], 50)
        response: Final = _chat(gateway, {"model": model})
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-attempted-retries"] == "1"
        assert response.headers["x-litellm-max-retries"] == "50"


def test_request_fallbacks_reroute_after_a_connection_failure(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        primary: Final = scenario.model(api_base=UNREACHABLE_API_BASE)
        fallback: Final = _unique_model(scenario)
        response: Final = _chat(gateway, {"model": primary, "fallbacks": [fallback.name]})
        assert response.status_code == 200, response.text
        assert _body(response)["model"] == fallback.upstream
        assert response.headers["x-litellm-model-id"] == fallback.deployment_id
        assert response.headers["x-litellm-attempted-fallbacks"] == "1"


def test_model_level_timeout_is_reported_on_a_timed_out_request(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, wire_server(_slow_reply) as wire:
        response: Final = _chat(gateway, {"model": scenario.model(api_base=wire.url, timeout=1)})
        assert response.status_code == 408, response.text
        assert response.headers["x-litellm-timeout"] == "1.0"


def test_request_timeout_header_overrides_the_model_timeout(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, wire_server(_slow_reply) as wire:
        response: Final = _chat(
            gateway,
            {"model": scenario.model(api_base=wire.url, timeout=1)},
            headers={"x-litellm-timeout": "0.001"},
        )
        assert response.status_code == 408, response.text
        assert response.headers["x-litellm-timeout"] == "0.001"


def test_failing_model_traffic_does_not_starve_concurrent_good_requests(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        bad: Final = scenario.model(api_base=UNREACHABLE_API_BASE)
        good: Final = scenario.model()
        for _ in range(2):
            bad_calls, good_calls = _concurrent_round(gateway, bad, good)
            assert good_calls == [200] * CONCURRENT_REQUESTS
            assert 200 not in bad_calls


def test_rpm_limited_deployment_rejects_a_second_call_while_the_first_is_in_flight(gateway: Gateway) -> None:
    release: Final = threading.Event()
    with gateway.scenario() as scenario, wire_server(_held_reply(release)) as wire, ThreadPoolExecutor(1) as pool:
        model: Final = scenario.model(api_base=wire.url, rpm=1)
        first: Final = pool.submit(_chat, gateway, {"model": model})
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            second: Final = _chat(gateway, {"model": model})
            assert second.status_code == 429, second.text
            assert wire.received.qsize() == 1
        finally:
            release.set()
        assert first.result().status_code == 200
        assert len(wire.drain()) == 1


def test_model_group_with_two_deployments_serves_from_both(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        first_id: Final = f"integration-{uuid.uuid4().hex}"
        model: Final = scenario.model(model_info={"id": first_id})
        second_id: Final = _deployment(gateway, scenario, model)
        served_by: Final = {_served_model_id(gateway, model) for _ in range(DISTRIBUTION_REQUESTS)}
        assert served_by == {first_id, second_id}


def test_unlisted_provider_model_resolves_through_a_wildcard_deployment(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        prefix: Final = f"integration{uuid.uuid4().hex}"
        _deployment(gateway, scenario, f"{prefix}/*", "openai/*")
        response: Final = _chat(gateway, {"model": f"{prefix}/gpt-4o-mini"})
        assert response.status_code == 200, response.text
        assert _content(response)


def test_comma_separated_models_fan_out_to_one_response_per_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        first: Final = _unique_model(scenario)
        second: Final = _unique_model(scenario)
        response: Final = _chat(gateway, {"model": f"{first.name},{second.name}"})
        assert response.status_code == 200, response.text
        replies: Final = JSON_OBJECTS.validate_json(response.content)
        assert len(replies) == 2, replies
        assert {string_value(reply["model"]) for reply in replies} == {first.upstream, second.upstream}
