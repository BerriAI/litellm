import uuid
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from pydantic import JsonValue


def test_an_exhausted_key_is_refused_inference_but_can_still_read_its_own_info(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], max_budget=0.06)
        assert (
            object_value(gateway.chat(model, key=key, text=f"spend {uuid.uuid4().hex}")["usage"])["total_tokens"] == 40
        )
        digest: Final = sha256(key.encode()).hexdigest()
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.06,
            seconds=70,
        )
        upstream.get("/__observations").raise_for_status()
        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"over budget {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert denied.status_code == 422, denied.text
        error: Final = denied.json()["error"]
        assert error["type"] == "budget_exceeded"
        assert "Budget has been exceeded!" in error["message"]
        assert upstream.get("/__observations").json()["requests"] == []
        info: Final = gateway.request("GET", "/key/info", key=key, params={"key": key})
        assert info.status_code == 200, info.text
        own: Final = object_value(info.json()["info"])
        assert float(str(own["spend"])) == pytest.approx(0.06)
        assert own["max_budget"] == 0.06


def _bounded_chat(gateway: Gateway, model: str, key: str, content: str | None = None) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": content or f"key recovery {uuid.uuid4().hex}"}],
        },
        key=key,
    )


def _observed(upstream: httpx.Client) -> list[JsonValue]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = object_value(response.json())["requests"]
    assert isinstance(requests, list), response.text
    return requests


def _upstream_chat(content: str) -> dict[str, JsonValue]:
    return {
        "path": "/v1/chat/completions",
        "authorization": "Bearer integration-provider-key",
        "body": {"model": "gpt-4o-mini", "max_tokens": 20, "messages": [{"role": "user", "content": content}]},
        "method": "POST",
        "api_key": "",
    }


def _served_chat(gateway: Gateway, upstream: httpx.Client, model: str, key: str) -> None:
    content: Final = f"tier served {uuid.uuid4().hex}"
    _observed(upstream)
    served: Final = _bounded_chat(gateway, model, key, content)
    assert served.status_code == 200, served.text
    assert float(served.headers["x-litellm-response-cost"]) == 0.06, dict(served.headers)
    assert _observed(upstream) == [_upstream_chat(content)], served.text


def test_raising_a_spent_keys_budget_restores_serving(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], max_budget=0.06)
        first: Final = _bounded_chat(gateway, model, key)
        assert first.status_code == 200, first.text
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.06,
            seconds=70,
        )
        eventually(lambda: _bounded_chat(gateway, model, key), lambda response: response.status_code != 200, seconds=30)
        upstream.get("/__observations").raise_for_status()
        denied: Final = _bounded_chat(gateway, model, key)
        assert denied.status_code == 422, denied.text
        assert object_value(denied.json()["error"])["type"] == "budget_exceeded"
        assert upstream.get("/__observations").json()["requests"] == []
        gateway.post("/key/update", {"key": key, "max_budget": 1.0})
        served: Final = tuple(_bounded_chat(gateway, model, key) for _ in range(3))
        assert [response.status_code for response in served] == [200, 200, 200], [response.text for response in served]
        assert len(upstream.get("/__observations").json()["requests"]) == 3


def _denied_with_budget_exceeded(gateway: Gateway, model: str, key: str) -> None:
    denied: Final = _bounded_chat(gateway, model, key)
    assert denied.status_code == 422, denied.text
    assert object_value(denied.json()["error"])["type"] == "budget_exceeded"


@pytest.mark.timeout(149)
def test_raising_a_spent_tier_budget_restores_serving(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        budget_id: Final = scenario.budget(budget_id=f"tier-{uuid.uuid4().hex}", max_budget=0.06, budget_duration="30d")
        key: Final = scenario.key(models=[model], budget_id=budget_id)
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert info["budget_id"] == budget_id, info
        tier_rows: Final = read_rows(
            'SELECT max_budget, budget_duration FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
        )
        assert len(tier_rows) == 1
        assert float(str(tier_rows[0]["max_budget"])) == 0.06
        assert tier_rows[0]["budget_duration"] == "30d"
        _served_chat(gateway, upstream, model, key)
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.06,
            seconds=70,
        )
        upstream.get("/__observations").raise_for_status()
        _denied_with_budget_exceeded(gateway, model, key)
        assert upstream.get("/__observations").json()["requests"] == []
        gateway.post("/budget/update", {"budget_id": budget_id, "max_budget": 1.0})
        updated_rows: Final = read_rows(
            'SELECT max_budget FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
        )
        assert float(str(updated_rows[0]["max_budget"])) == 1.0, updated_rows
        restored_content: Final = f"tier restored {uuid.uuid4().hex}"
        served: Final = eventually(
            lambda: _bounded_chat(gateway, model, key, restored_content),
            lambda response: response.status_code == 200,
            seconds=125,
        )
        assert served.status_code == 200, served.text
        assert _observed(upstream) == [_upstream_chat(restored_content)], served.text


@pytest.mark.timeout(149)
def test_lowering_a_tier_budget_below_spend_blocks_the_key(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        budget_id: Final = scenario.budget(
            budget_id=f"tier-{uuid.uuid4().hex}", max_budget=1000.0, budget_duration="30d"
        )
        key: Final = scenario.key(models=[model], budget_id=budget_id)
        _served_chat(gateway, upstream, model, key)
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.06,
            seconds=30,
        )
        upstream.get("/__observations").raise_for_status()
        probe: Final = gateway.request("POST", "/v1/chat/completions", {"model": model}, key=key)
        assert probe.status_code == 400, probe.text
        assert object_value(probe.json()["error"]) == {
            "message": "/chat/completions: Missing required parameter: 'messages'.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "400",
        }, probe.text
        assert upstream.get("/__observations").json()["requests"] == []
        gateway.post("/budget/update", {"budget_id": budget_id, "max_budget": 0.01})
        updated_rows: Final = read_rows(
            'SELECT max_budget FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
        )
        assert len(updated_rows) == 1
        assert float(str(updated_rows[0]["max_budget"])) == 0.01, updated_rows
        upstream.get("/__observations").raise_for_status()
        lowered_probe: Final = eventually(
            lambda: gateway.request("POST", "/v1/chat/completions", {"model": model}, key=key),
            lambda response: (
                response.status_code == 422 and object_value(response.json()["error"])["type"] == "budget_exceeded"
            ),
            seconds=125,
        )
        assert "Max budget: 0.01" in lowered_probe.text, lowered_probe.text
        denied: Final = _bounded_chat(gateway, model, key)
        assert denied.status_code == 422, denied.text
        assert object_value(denied.json()["error"])["type"] == "budget_exceeded", denied.text
        assert "Max budget: 0.01" in denied.text, denied.text
        assert upstream.get("/__observations").json()["requests"] == []
