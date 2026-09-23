from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value

PAYLOAD_MEGABYTES: Final = 16
BURST_REQUESTS: Final = 8
RETAINED_MEGABYTES_ALLOWED: Final = PAYLOAD_MEGABYTES // 2


def _resident_megabytes(gateway: Gateway) -> float:
    memory: Final = object_value(gateway.get("/debug/memory/summary")["memory"])
    return float(str(memory["ram_usage_mb"]))


@pytest.mark.covers("other.routing.memory.large_request_payloads_are_released_after_the_request_ends")
def test_worker_memory_settles_after_a_burst_of_large_chat_requests(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        warm: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "warm"}]}
        )
        assert warm.status_code == 200, warm.text
        baseline: Final = _resident_megabytes(gateway)
        filler: Final = "x" * (PAYLOAD_MEGABYTES * 1024 * 1024)
        for index in range(BURST_REQUESTS):
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"{filler}{index}"}]},
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["role"] == "assistant", response.text
        settled: Final = eventually(
            lambda: _resident_megabytes(gateway),
            lambda resident: resident - baseline < RETAINED_MEGABYTES_ALLOWED,
            seconds=30,
            return_last_on_timeout=True,
        )
        assert settled - baseline < RETAINED_MEGABYTES_ALLOWED, (
            f"worker RSS grew by {settled - baseline:.1f} MB after {BURST_REQUESTS} requests of {PAYLOAD_MEGABYTES} MB"
        )
