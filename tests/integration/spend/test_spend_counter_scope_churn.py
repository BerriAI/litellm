import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from pydantic import TypeAdapter

GENERIC_CACHE_SIZE: Final = 200
CHURN_SCOPES: Final = 250


def _chat(candidate: Gateway, model: str, key: str, user: str) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"scope churn {uuid.uuid4().hex}"}], "user": user},
        key=key,
    )


@pytest.mark.covers("quota_management.budget.spend_counter.survives_scope_churn")
@pytest.mark.timeout(180)
def test_over_budget_key_stays_blocked_while_hundreds_of_end_users_take_traffic(
    gateway: Gateway, tmp_path: Path
) -> None:
    shared: Final = TypeAdapter(dict[str, object]).validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    general_settings: Final = TypeAdapter(dict[str, object]).validate_python(shared["general_settings"])
    path: Final = tmp_path / "in-memory-counters.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [],
                "general_settings": {**general_settings, "proxy_batch_write_at": 600},
                "router_settings": shared["router_settings"],
            }
        )
    )
    run: Final = uuid.uuid4().hex
    with (
        owned_proxy(gateway, tmp_path, {}, config=path, remove_environment=("REDIS_HOST", "REDIS_PORT")) as candidate,
        candidate.scenario() as scenario,
        httpx.Client(base_url=candidate.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        probe: Final = scenario.key(models=[model], max_budget=0.06)
        churn: Final = scenario.key(models=[model])

        def _churn_status(index: int) -> int:
            return _chat(candidate, model, churn, f"churn-{run}-{index}").status_code

        first: Final = _chat(candidate, model, probe, f"probe-{run}")
        assert first.status_code == 200 and first.json()["usage"]["total_tokens"] == 40, first.text
        blocked: Final = _chat(candidate, model, probe, f"probe-{run}")
        assert blocked.status_code == 422 and blocked.json()["error"]["type"] == "budget_exceeded", blocked.text
        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses: Final = tuple(pool.map(_churn_status, range(CHURN_SCOPES)))
        assert statuses.count(200) > GENERIC_CACHE_SIZE, statuses
        upstream.get("/__observations").raise_for_status()
        after: Final = _chat(candidate, model, probe, f"probe-{run}")
        assert after.status_code == 422 and after.json()["error"]["type"] == "budget_exceeded", (
            f"over-budget key admitted after {statuses.count(200)} other scopes took traffic: {after.status_code} {after.text}"
        )
        assert upstream.get("/__observations").json()["requests"] == []
