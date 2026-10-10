import os
import uuid
from typing import Final

import httpx
from integration._support.client import Gateway, object_value, string_value
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse
from pydantic import JsonValue

_DECISIONS_PROBE_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "model": "pplx-decider-v1-27b",
        "answers": {"reachable": {"type": "noul", "noul": 1.0}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
)
_CONFIGURED_PROBE_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "model": "jev-custom",
        "answers": {"alive": {"type": "choice", "choice": "yes", "confidence": 0.9, "probabilities": {"yes": 0.9}}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
)
_VLLM_PROBE_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "model": "Qwen/Qwen3-0.6B",
        "answers": {
            "reachable": {"type": "choice", "choice": "yes", "confidence": 1.0, "probabilities": {"yes": 1.0, "no": 0.0}}
        },
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
)
_STRANDS_PROBE_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "model": "strands-decider-2B-hobson-v19",
        "answers": {"reachable": {"type": "noul", "noul": 1.0}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
)
_CONFIGURED_STATE: Final[dict[str, JsonValue]] = {"ticket": "health probe"}
_CONFIGURED_QUESTIONS: Final[dict[str, JsonValue]] = {
    "alive": {"type": "choice", "criteria": {"yes": "the service answers", "no": "the service is down"}}
}


def test_health_check_of_a_model_added_through_the_api_calls_its_upstream_and_reports_it_healthy(
    gateway: Gateway,
) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        provider_model: Final = f"health-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}")
        key: Final = scenario.key(models=[model])
        listed: Final = gateway.request("GET", "/v2/model/info", key=key, params={"model": model})
        assert listed.status_code == 200, listed.text
        assert [entry["model_name"] for entry in listed.json()["data"]] == [model]
        assert (
            object_value(gateway.chat(model, key=key, text=f"health {uuid.uuid4().hex}")["usage"])["total_tokens"] == 40
        )
        upstream.get("/__observations").raise_for_status()
        health: Final = gateway.request("GET", "/health", params={"model": model})
        assert health.status_code == 200, health.text
        report: Final = health.json()
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert [(endpoint["model"], endpoint["api_base"]) for endpoint in report["healthy_endpoints"]] == [
            (f"openai/{provider_model}", f"{gateway.upstream_url}/v1")
        ]
        assert [request["body"]["model"] for request in upstream.get("/__observations").json()["requests"]] == [
            provider_model
        ]


def _health_report(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    health: Final = gateway.request("GET", "/health", params={"model": model})
    assert health.status_code == 200, health.text
    return health.json()


def _probes_sent_to(gateway: Gateway, handle: ScenarioHandle) -> list[tuple[str, JsonValue]]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        requests: Final = upstream.get("/__observations").json()["requests"]
    return [
        (string_value(request["path"]), request["body"])
        for request in map(object_value, requests)
        if string_value(request["path"]).startswith(f"/{handle.scenario_id}/")
    ]


def test_evaluation_mode_health_check_resolves_the_mode_from_the_cost_map_and_sends_the_default_probe(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(f"health-decisions-{uuid.uuid4().hex[:12]}", _DECISIONS_PROBE_REPLY)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(model="perplexity/pplx-decider-v1-27b", api_base=handle.api_base())
        report: Final = _health_report(gateway, model)
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert _probes_sent_to(gateway, handle) == [
            (
                f"/{handle.scenario_id}/v1/decisions",
                {
                    "model": "pplx-decider-v1-27b",
                    "state": os.environ.get("DEFAULT_HEALTH_CHECK_PROMPT", "test from litellm"),
                    "questions": {"reachable": {"type": "noul", "instructions": "Is the service reachable?"}},
                },
            )
        ]


def test_evaluation_mode_health_check_sends_the_configured_state_and_questions(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(f"health-decisions-{uuid.uuid4().hex[:12]}", _CONFIGURED_PROBE_REPLY)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="typesafe/jev-custom",
            api_base=handle.api_base(),
            model_info={
                "mode": "evaluation",
                "health_check_params": {"state": _CONFIGURED_STATE, "questions": _CONFIGURED_QUESTIONS},
            },
        )
        report: Final = _health_report(gateway, model)
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert _probes_sent_to(gateway, handle) == [
            (
                f"/{handle.scenario_id}/v1/systemone",
                {"model": "jev-custom", "state": _CONFIGURED_STATE, "questions": _CONFIGURED_QUESTIONS},
            )
        ]


def test_evaluation_mode_health_check_of_the_self_hosted_strands_model_resolves_the_mode_from_the_cost_map(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(f"health-decisions-{uuid.uuid4().hex[:12]}", _STRANDS_PROBE_REPLY)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="strands_decider/strands-decider-2B-hobson-v19", api_base=handle.api_base(), api_key=None
        )
        report: Final = _health_report(gateway, model)
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert _probes_sent_to(gateway, handle) == [
            (
                f"/{handle.scenario_id}/v1/systemone",
                {
                    "model": "strands-decider-2B-hobson-v19",
                    "state": os.environ.get("DEFAULT_HEALTH_CHECK_PROMPT", "test from litellm"),
                    "questions": {"reachable": {"type": "noul", "instructions": "Is the service reachable?"}},
                },
            )
        ]


def test_evaluation_mode_health_check_of_hosted_vllm_sends_a_choice_probe(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(f"health-decisions-{uuid.uuid4().hex[:12]}", _VLLM_PROBE_REPLY)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="hosted_vllm/Qwen/Qwen3-0.6B",
            api_base=handle.api_base(),
            model_info={"mode": "evaluation"},
        )
        report: Final = _health_report(gateway, model)
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert _probes_sent_to(gateway, handle) == [
            (
                f"/{handle.scenario_id}/v1/systemone",
                {
                    "model": "Qwen/Qwen3-0.6B",
                    "state": os.environ.get("DEFAULT_HEALTH_CHECK_PROMPT", "test from litellm"),
                    "questions": {
                        "reachable": {
                            "type": "choice",
                            "instructions": "Is the service reachable?",
                            "criteria": {"yes": None, "no": None},
                        }
                    },
                },
            )
        ]


_DATABRICKS_ENDPOINT_PROBE_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "model": "databricks-openjev-qwen35-4b",
        "answers": {"reachable": {"type": "noul", "noul": 1.0}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
)


def test_evaluation_mode_health_check_of_a_databricks_serving_endpoint_resolves_the_mode_from_the_cost_map(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(f"health-decisions-{uuid.uuid4().hex[:12]}", _DATABRICKS_ENDPOINT_PROBE_REPLY)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="databricks/databricks-openjev-qwen35-4b",
            api_base=handle.api_base(),
            api_key="synthetic-databricks-key",
        )
        listed: Final = gateway.request("GET", "/v2/model/info", params={"model": model})
        assert listed.status_code == 200, listed.text
        assert [object_value(object_value(entry)["model_info"])["mode"] for entry in listed.json()["data"]] == [
            "evaluation"
        ], listed.text
        report: Final = _health_report(gateway, model)
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert _probes_sent_to(gateway, handle) == [
            (
                f"/{handle.scenario_id}/databricks-openjev-qwen35-4b/invocations",
                {
                    "model": "databricks-openjev-qwen35-4b",
                    "state": os.environ.get("DEFAULT_HEALTH_CHECK_PROMPT", "test from litellm"),
                    "questions": {"reachable": {"type": "noul", "instructions": "Is the service reachable?"}},
                },
            )
        ]


_DATABRICKS_AI_DECIDE_PROBE_REPLY: Final = JsonResponse(
    content_type="application/json",
    body={
        "response": {"answers": {"reachable": {"type": "noul", "probability": 1.0}}},
        "metadata": {"version": "1.0"},
    },
)


def test_evaluation_mode_health_check_of_databricks_ai_decide_resolves_the_mode_from_the_cost_map(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(f"health-decisions-{uuid.uuid4().hex[:12]}", _DATABRICKS_AI_DECIDE_PROBE_REPLY)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="databricks/ai_decide",
            api_base=handle.api_base(),
            api_key="synthetic-databricks-key",
        )
        listed: Final = gateway.request("GET", "/v2/model/info", params={"model": model})
        assert listed.status_code == 200, listed.text
        assert [object_value(object_value(entry)["model_info"])["mode"] for entry in listed.json()["data"]] == [
            "evaluation"
        ], listed.text
        report: Final = _health_report(gateway, model)
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        assert _probes_sent_to(gateway, handle) == [
            (
                f"/{handle.scenario_id}/api/2.0/ai-functions/ai-decide",
                {
                    "state": os.environ.get("DEFAULT_HEALTH_CHECK_PROMPT", "test from litellm"),
                    "questions": {"reachable": {"type": "noul", "instructions": "Is the service reachable?"}},
                },
            )
        ]
