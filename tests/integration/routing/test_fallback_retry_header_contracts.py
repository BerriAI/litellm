import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final, Literal

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, object_value, string_value
from tests.integration._support.openai_wire import chat_reply
from tests.integration._support.wire import Reply, Request, wire_server

UNREACHABLE_API_BASE: Final = "http://127.0.0.1:9/v1"
END_USER_REQUESTS: Final = 10
CONCURRENT_REQUESTS: Final = 25
DISTRIBUTION_REQUESTS: Final = 20
SLOW_UPSTREAM_SECONDS: Final = 3
CUSTOM_FALLBACK_TEXT: Final = "custom fallback prompt"
Caller = Literal["virtual-key", "master-key"]


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


def _content(response: httpx.Response) -> str:
    choices: Final = object_value(response.json())["choices"]
    assert isinstance(choices, list) and choices, response.text
    return string_value(object_value(object_value(choices[0])["message"])["content"])


def _scripted_model(gateway: Gateway, scenario: Scenario, statuses: list[int], **parameters: JsonValue) -> str:
    upstream_model: Final = f"integration-{uuid.uuid4().hex}"
    script_url: Final = f"{gateway.upstream_url}/__scripts/{upstream_model}"
    configured: Final = httpx.post(script_url, json={"statuses": statuses})
    assert configured.status_code == 200, configured.text
    scenario.cleanups.callback(httpx.delete, script_url)
    return scenario.model(model=f"openai/{upstream_model}", **parameters)


def _unique_model(scenario: Scenario, **parameters: JsonValue) -> tuple[str, str]:
    upstream_model: Final = f"integration-{uuid.uuid4().hex}"
    return scenario.model(model=f"openai/{upstream_model}", **parameters), upstream_model


def _slow_reply(_: Request) -> Reply:
    time.sleep(SLOW_UPSTREAM_SECONDS)
    return chat_reply("chatcmpl-slow", "gpt-4o-mini", "late", stream=False)


def _fallback_reply(_: Request) -> Reply:
    return chat_reply("chatcmpl-fallback", "gpt-4o-mini", "served by fallback", stream=False)


def _deployment(gateway: Gateway, scenario: Scenario, model_name: str, model: str = "openai/gpt-4o-mini") -> None:
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
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))


@pytest.mark.parametrize("caller", ["virtual-key", "master-key"])
def test_end_user_budget_tpm_limit_rate_limits_their_requests(gateway: Gateway, caller: Caller) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        budget: Final = scenario.budget(tpm_limit=2)
        end_user: Final = f"integration-{uuid.uuid4().hex}"
        gateway.post("/end_user/new", {"user_id": end_user, "budget_id": budget})
        scenario.cleanups.callback(gateway.post, "/end_user/delete", {"user_ids": [end_user]})
        key: Final = scenario.key(models=[model]) if caller == "virtual-key" else gateway.key
        control: Final = _chat(gateway, {"model": model, "user": f"integration-{uuid.uuid4().hex}"}, key=key)
        assert control.status_code == 200, control.text
        statuses: Final = [
            _chat(gateway, {"model": model, "user": end_user}, key=key).status_code for _ in range(END_USER_REQUESTS)
        ]
        assert statuses.count(200) < 5, statuses
        assert set(statuses) <= {200, 429}, statuses


def test_client_fallbacks_reach_an_allowed_model_and_name_a_denied_one(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        primary: Final = scenario.model(api_base=UNREACHABLE_API_BASE)
        fallback, fallback_upstream = _unique_model(scenario)
        body: Final[dict[str, JsonValue]] = {"model": primary, "fallbacks": [fallback]}
        served: Final = _chat(gateway, body, key=scenario.key(models=[primary, fallback]))
        assert served.status_code == 200, served.text
        assert object_value(served.json())["model"] == fallback_upstream
        assert _content(served)
        denied: Final = _chat(gateway, body, key=scenario.key(models=[primary]))
        assert denied.status_code == 403, denied.text
        assert fallback in denied.text


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
        forwarded: Final = [object_value(json.loads(request.body))["messages"] for request in wire.drain()]
        assert forwarded == [custom]
        denied: Final = _chat(gateway, body, key=scenario.key(models=[primary]))
        assert denied.status_code == 403, denied.text
        assert fallback in denied.text
        assert wire.drain() == ()


def test_rate_limited_deployment_is_retried_and_reports_retry_counts(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _scripted_model(gateway, scenario, [429, 200], num_retries=50)
        response: Final = _chat(gateway, {"model": model})
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-attempted-retries"] == "1"
        assert response.headers["x-litellm-max-retries"] == "50"


def test_request_fallbacks_reroute_after_a_connection_failure(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        primary: Final = scenario.model(api_base=UNREACHABLE_API_BASE)
        fallback, fallback_upstream = _unique_model(scenario)
        response: Final = _chat(gateway, {"model": primary, "fallbacks": [fallback]})
        assert response.status_code == 200, response.text
        assert object_value(response.json())["model"] == fallback_upstream
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
            with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS * 2) as pool:
                bad_calls: list[int] = list(
                    pool.map(lambda _: _chat(gateway, {"model": bad}).status_code, range(CONCURRENT_REQUESTS))
                )
                good_calls: list[int] = list(
                    pool.map(lambda _: _chat(gateway, {"model": good}).status_code, range(CONCURRENT_REQUESTS))
                )
            assert good_calls == [200] * CONCURRENT_REQUESTS
            assert 200 not in bad_calls


def test_rpm_limited_deployment_rejects_a_parallel_second_call(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(rpm=1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses: Final = sorted(pool.map(lambda _: _chat(gateway, {"model": model}).status_code, range(2)))
        assert 429 in statuses, statuses


def test_model_group_with_two_deployments_serves_from_both(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        _deployment(gateway, scenario, model)
        served_by: Final = set[str]()
        for _ in range(DISTRIBUTION_REQUESTS):
            response = _chat(gateway, {"model": model})
            assert response.status_code == 200, response.text
            served_by.add(response.headers["x-litellm-model-id"])
        assert len(served_by) == 2, served_by


def test_unlisted_provider_model_resolves_through_a_wildcard_deployment(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        prefix: Final = f"integration{uuid.uuid4().hex}"
        _deployment(gateway, scenario, f"{prefix}/*", "openai/*")
        response: Final = _chat(gateway, {"model": f"{prefix}/gpt-4o-mini"})
        assert response.status_code == 200, response.text
        assert _content(response)


def test_comma_separated_models_fan_out_to_one_response_per_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        first, first_upstream = _unique_model(scenario)
        second, second_upstream = _unique_model(scenario)
        response: Final = _chat(gateway, {"model": f"{first},{second}"})
        assert response.status_code == 200, response.text
        replies: Final = response.json()
        assert isinstance(replies, list) and len(replies) == 2, replies
        assert {object_value(reply)["model"] for reply in replies} == {first_upstream, second_upstream}
