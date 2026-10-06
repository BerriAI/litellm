import json
import uuid
from collections.abc import Callable
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.routing.retries.several_attempts_reach_success_without_hidden_retries", "other.routing.errors.nonretryable_and_exhausted_failures_remain_errors")
def test_retry_counts_and_public_errors_match_actual_provider_attempts(gateway: Gateway) -> None:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream, gateway.scenario() as scenario:
        original: Final = object_value(gateway.get("/router/settings")["current_values"])["num_retries"]
        provider_model: Final = "errors-" + uuid.uuid4().hex
        model: Final = scenario.model(model=f"openai/{provider_model}", input_cost_per_token=0, output_cost_per_token=0)

        def remove() -> None:
            response: Final = upstream.delete(f"/__scripts/{provider_model}")
            assert response.status_code in (200, 404)
            assert upstream.get(f"/__scripts/{provider_model}").status_code == 404

        scenario.cleanups.callback(remove)
        try:
            for index, (retries, statuses, status, attempts) in enumerate(((2, [500, 500, 200], 200, 3), (2, [400, 200], 400, 1), (1, [429, 429, 200], 429, 2), (1, [500, 500, 200], 500, 2))):
                gateway.post("/config/update", {"router_settings": {"num_retries": retries}})
                assert object_value(gateway.get("/router/settings")["current_values"])["num_retries"] == retries
                upstream.post(f"/__scripts/{provider_model}", json={"statuses": statuses}).raise_for_status()
                upstream.get("/__observations").raise_for_status()
                response: Final = gateway.request("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": f"{provider_model} {index}"}]})
                assert response.status_code == status, response.text
                requests: Final = upstream.get("/__observations").json()["requests"]
                assert len(requests) == attempts
                assert all(request["body"]["model"] == provider_model for request in requests)
                assert upstream.get(f"/__scripts/{provider_model}").json()["remaining"] == statuses[attempts:]
                if status == 200:
                    assert response.json()["usage"]["total_tokens"] == 40
                else:
                    error: Final = response.json()["error"]
                    assert isinstance(error["message"], str) and "Controlled provider failure" in error["message"]
                    assert str(error["code"]) == str(status)
                    assert error["type"] == {400: "invalid_request_error", 429: "throttling_error", 500: "internal_server_error"}[status]
                    assert error["param"] is None
                    assert "Traceback" not in response.text and "File \"" not in response.text
        finally:
            gateway.post("/config/update", {"router_settings": {"num_retries": original}})
            assert object_value(gateway.get("/router/settings")["current_values"])["num_retries"] == original


@pytest.mark.covers("other.routing.fallback.loaded_configuration_selects_only_permitted_target")
def test_loaded_fallback_selects_expected_deployment_and_keeps_response_identity(tmp_path: Path) -> None:
    from litellm import Router

    def respond(request: Request) -> Reply:
        model: Final = json.loads(request.body)["model"]
        assert model in {"primary-wire", "fallback-wire", "unrelated-wire"}
        if model == "primary-wire":
            return Reply(status=500, body=b'{"error":{"message":"synthetic primary unavailable","type":"api_error","code":"500"}}')
        return Reply(body=json.dumps({"id": "response-" + model, "object": "chat.completion", "created": 1, "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": "served " + model}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}}).encode())

    with wire_server(respond) as wire:
        path: Final = tmp_path / "fallback.yaml"
        path.write_text(yaml.safe_dump({"model_list": [{"model_name": alias, "litellm_params": {"model": "openai/" + upstream, "api_key": "synthetic-routing-key", "api_base": wire.url + "/v1"}} for alias, upstream in (("primary", "primary-wire"), ("fallback", "fallback-wire"), ("unrelated", "unrelated-wire"))], "router_settings": {"num_retries": 0, "disable_cooldowns": True, "fallbacks": [{"primary": ["fallback"]}]}}))
        loaded: Final = yaml.safe_load(path.read_text())
        router: Final = Router(model_list=loaded["model_list"], **loaded["router_settings"])
        try:
            result: Final = router.completion(model="primary", messages=[{"role": "user", "content": "fallback control"}])
            assert result.id == "response-fallback-wire"
            assert result.choices[0].message.content == "served fallback-wire"
            assert result.choices[0].finish_reason == "stop"
            assert result.usage.prompt_tokens == 11 and result.usage.completion_tokens == 4
            assert tuple(json.loads(request.body)["model"] for request in wire.drain()) == ("primary-wire", "fallback-wire")
            control: Final = router.completion(model="unrelated", messages=[{"role": "user", "content": "independent route"}])
            assert control.id == "response-unrelated-wire"
            assert tuple(json.loads(request.body)["model"] for request in wire.drain()) == ("unrelated-wire",)
        finally:
            router.reset()


@pytest.mark.covers("other.routing.alias_update.persisted_target_changes_only_selected_route")
def test_saved_deployment_target_update_changes_wire_and_preserves_control(gateway: Gateway) -> None:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream, gateway.scenario() as scenario:
        prefix: Final = "target-" + uuid.uuid4().hex
        model: Final = scenario.model(model="openai/" + prefix + "-first", input_cost_per_token=0, output_cost_per_token=0)
        other: Final = scenario.model(model="openai/" + prefix + "-control", input_cost_per_token=0, output_cost_per_token=0)
        target: Final = next(entry for entry in gateway.get("/model/info")["data"] if entry["model_name"] == model)
        for generation, suffix in enumerate(("first", "second")):
            if generation:
                response: Final = gateway.request("PATCH", f"/model/{target['model_info']['id']}/update", {"litellm_params": {"model": "openai/" + prefix + "-second"}})
                assert response.status_code == 200, response.text
            upstream.get("/__observations").raise_for_status()
            for alias in (model, other):
                assert gateway.chat(alias, text=f"{prefix} generation {generation}")["usage"]["total_tokens"] == 40
            requests: Final = upstream.get("/__observations").json()["requests"]
            assert [request["body"]["model"] for request in requests] == [prefix + "-" + suffix, prefix + "-control"]


@pytest.mark.parametrize(
    (
        "case",
        "primary_error",
        "body_flag",
        "expected_sequence",
        "expected_model",
        "expected_status",
        "expected_attempted_fallbacks",
    ),
    (
        ("context", "context", None, ("primary-up", "big-up"), "big-up", 200, "1"),
        ("content-policy", "content-policy", None, ("primary-up", "safe-up"), "safe-up", 200, "1"),
        ("server-error", "server-error", None, ("primary-up", "general-up"), "general-up", 200, "1"),
        ("disabled-context", "context", "disable_fallbacks", ("primary-up",), "primary-up", 400, None),
        ("mock-context", None, "mock_testing_context_fallbacks", ("big-up",), "big-up", 200, "1"),
        ("mock-content-policy", None, "mock_testing_content_policy_fallbacks", ("safe-up",), "safe-up", 200, "1"),
        ("mock-server-error", None, "mock_testing_fallbacks", ("general-up",), "general-up", 200, "1"),
    ),
)
def test_health_routing_fallbacks_preserve_error_specific_targets_and_request_controls(
    gateway: Gateway,
    tmp_path: Path,
    case: str,
    primary_error: str | None,
    body_flag: str | None,
    expected_sequence: tuple[str, ...],
    expected_model: str,
    expected_status: int,
    expected_attempted_fallbacks: str | None,
) -> None:
    prompt: Final = f"health-routing-{case}-{uuid.uuid4().hex}"
    call_order: Final = SimpleQueue[str]()

    def respond(model: str) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "response-" + model,
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "served " + model},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )

    def handler(model: str) -> Callable[[Request], Reply]:
        def handle(request: Request) -> Reply:
            if request.method == "GET":
                return Reply(body=b'{"object":"list","data":[]}')
            actual_body: Final = json.loads(request.body)
            assert request.target == "/v1/chat/completions", request.target
            assert actual_body == {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }, request.body.decode()
            call_order.put(model)
            if model == "primary-up" and primary_error == "context":
                return Reply(
                    status=400,
                    body=b'{"error":{"message":"This model\'s maximum context length is 4096 tokens","type":"invalid_request_error","code":"context_length_exceeded"}}',
                )
            if model == "primary-up" and primary_error == "content-policy":
                return Reply(
                    status=400,
                    body=b'{"error":{"message":"synthetic content policy rejection","type":"invalid_request_error","code":"content_policy_violation"}}',
                )
            if model == "primary-up" and primary_error == "server-error":
                return Reply(
                    status=500,
                    body=b'{"error":{"message":"synthetic upstream failure","type":"api_error","code":"500"}}',
                )
            return respond(model)

        return handle

    with (
        wire_server(handler("primary-up")) as primary,
        wire_server(handler("big-up")) as big,
        wire_server(handler("safe-up")) as safe,
        wire_server(handler("general-up")) as general,
    ):
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            {
                "model_name": alias,
                "litellm_params": {
                    "model": f"openai/{upstream_model}",
                    "api_base": wire.url + "/v1",
                    "api_key": "synthetic-health-routing-key",
                },
            }
            for alias, upstream_model, wire in (
                ("primary", "primary-up", primary),
                ("big", "big-up", big),
                ("safe", "safe-up", safe),
                ("general", "general-up", general),
            )
        ]
        config["router_settings"] = {
            "num_retries": 0,
            "disable_cooldowns": True,
            "context_window_fallbacks": [{"primary": ["big"]}],
            "content_policy_fallbacks": [{"primary": ["safe"]}],
            "fallbacks": [{"primary": ["general"]}],
        }
        config["general_settings"] = {
            **config.get("general_settings", {}),
            "dangerously_allow_mock_testing_request_params": True,
        }
        path: Final = tmp_path / f"health-routing-{case}.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate:
            request_body: Final = {
                "model": "primary",
                "messages": [{"role": "user", "content": prompt}],
                **({body_flag: True} if body_flag is not None else {}),
            }
            response: Final = candidate.request("POST", "/v1/chat/completions", request_body)
            assert response.status_code == expected_status, response.text
            observed_order: Final = tuple(call_order.get_nowait() for _ in range(call_order.qsize()))
            assert observed_order == expected_sequence
            assert response.headers.get("x-litellm-attempted-fallbacks") == expected_attempted_fallbacks, response.text
            if expected_status == 400:
                assert response.json() == {
                    "error": {
                        "message": (
                            "litellm.ContextWindowExceededError: litellm.BadRequestError: "
                            "ContextWindowExceededError: OpenAIException - This model's maximum context length is "
                            "4096 tokens"
                        ),
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "400",
                    }
                }, response.text
            else:
                assert response.json() == {
                    "id": "response-" + expected_model,
                    "object": "chat.completion",
                    "created": 1,
                    "model": expected_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "served " + expected_model,
                                "provider_specific_fields": {"refusal": None},
                            },
                            "finish_reason": "stop",
                            "provider_specific_fields": {},
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                }, response.text
