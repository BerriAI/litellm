import uuid
from typing import Final

import httpx
from integration._support.client import Gateway, object_value


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
