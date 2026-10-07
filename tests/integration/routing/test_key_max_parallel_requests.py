import uuid
from typing import Final

import httpx
from integration._support.client import Gateway, object_value


def test_zero_parallel_slots_refuse_before_the_provider_and_one_slot_serves(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        blocked: Final = scenario.key(models=[model], max_parallel_requests=0)
        allowed: Final = scenario.key(models=[model], max_parallel_requests=1)
        upstream.get("/__observations").raise_for_status()
        refused: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"no slots {uuid.uuid4().hex}"}]},
            key=blocked,
        )
        assert refused.status_code == 429, refused.text
        assert upstream.get("/__observations").json()["requests"] == []
        served: Final = tuple(gateway.chat(model, key=allowed, text=f"one slot {uuid.uuid4().hex}") for _ in range(2))
        assert [object_value(response["usage"])["total_tokens"] for response in served] == [40, 40]
        assert len(upstream.get("/__observations").json()["requests"]) == 2
