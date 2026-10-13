import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import httpx
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows

_BURST: Final = 24


def test_burst_with_partial_upstream_failures_logs_each_success_once(gateway: Gateway) -> None:
    with (
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        gateway.scenario() as scenario,
    ):
        provider_model: Final = f"burst-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}", input_cost_per_token=0, output_cost_per_token=0)
        statuses: Final = [500] + [200, 200, 200] * (_BURST // 4 + 2)

        def remove_script() -> None:
            response: Final = upstream.delete(f"/__scripts/{provider_model}")
            assert response.status_code in (200, 404), response.text

        scenario.cleanups.callback(remove_script)
        configured: Final = upstream.post(f"/__scripts/{provider_model}", json={"statuses": statuses})
        assert configured.status_code == 200, configured.text
        upstream.get("/__observations").raise_for_status()

        def attempt(index: int) -> httpx.Response:
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"burst {index}"}]},
            )

        with ThreadPoolExecutor(max_workers=_BURST) as pool:
            responses: Final = tuple(pool.map(attempt, range(_BURST)))

        succeeded: Final = tuple(response.json()["id"] for response in responses if response.status_code == 200)
        assert len(succeeded) > 0, [response.status_code for response in responses]
        assert len(set(succeeded)) == len(succeeded), "duplicate response id in burst"
        assert all(response.status_code in (200, 429, 500) for response in responses), [
            response.status_code for response in responses
        ]

        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)',
                (list(succeeded),),
            ),
            lambda values: len(values) == len(succeeded),
            seconds=90,
        )
        landed: Final = [row["request_id"] for row in rows]
        assert sorted(landed) == sorted(succeeded), "a successful burst id did not land exactly once"
