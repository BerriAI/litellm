import uuid
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows


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


def _bounded_chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": f"key recovery {uuid.uuid4().hex}"}],
        },
        key=key,
    )


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
