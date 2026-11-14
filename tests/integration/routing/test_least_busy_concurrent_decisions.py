import json
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

pytestmark = pytest.mark.timeout(240)

_GROUP: Final = "least-busy-decider"
_DEPLOYMENTS: Final = ("least-busy-decider-a", "least-busy-decider-b")
_BURST: Final = 4
_REPLY: Final = json.dumps(
    {
        "model": "Qwen/Qwen3-0.6B",
        "answers": {"billing": {"type": "noul", "noul": 0.91}},
        "usage": {"input_tokens": 367, "output_tokens": 3},
    }
).encode()
_DECISION: Final = {
    "model": _GROUP,
    "input": "I was charged twice for order 4417.",
    "questions": [{"type": "predicate", "name": "billing", "instructions": "Is this a billing issue?"}],
}


def _held_until_the_burst_arrives(arrived: threading.Barrier) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        arrived.wait()
        return Reply(body=_REPLY)

    return respond


def _write_config(path: Path, provider_url: str) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": _GROUP,
                        "litellm_params": {"model": "hosted_vllm/Qwen/Qwen3-0.6B", "api_base": provider_url},
                        "model_info": {"id": deployment_id},
                    }
                    for deployment_id in _DEPLOYMENTS
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_model_info_refresh": True,
                },
                "router_settings": {"routing_strategy": "least-busy", "num_retries": 0, "disable_cooldowns": True},
            }
        )
    )
    return path


def _decide(proxy: Gateway) -> httpx.Response:
    return proxy.client.post(
        "/v1/decisions", json=_DECISION, headers={"Authorization": f"Bearer {proxy.key}"}, timeout=120
    )


def test_least_busy_spreads_a_burst_of_decisions_calls_across_the_group(tmp_path: Path) -> None:
    arrived: Final = threading.Barrier(_BURST, timeout=60)
    with (
        wire_server(_held_until_the_burst_arrives(arrived)) as upstream,
        gateway_from_environment() as rig,
        owned_proxy(
            rig,
            tmp_path,
            {"STORE_MODEL_IN_DB": "False"},
            config=_write_config(tmp_path / "least-busy.yaml", upstream.url),
        ) as proxy,
        ThreadPoolExecutor(max_workers=_BURST) as pool,
    ):
        responses: Final = tuple(pool.map(lambda _: _decide(proxy), range(_BURST)))
    assert [response.status_code for response in responses] == [200] * _BURST, [r.text for r in responses]
    assert Counter(response.headers["x-litellm-model-id"] for response in responses) == Counter(
        {deployment_id: _BURST // len(_DEPLOYMENTS) for deployment_id in _DEPLOYMENTS}
    )
