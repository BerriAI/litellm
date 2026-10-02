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
