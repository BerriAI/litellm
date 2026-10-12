import json
import re
import uuid
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, object_value
from tests.integration._support.process import owned_proxy

SAMPLES: Final = 4
REQUESTS_PER_INTERVAL: Final = 5
LEAK_MIN_NET_GROWTH: Final = 5
LEAK_MIN_GROWING_INTERVALS: Final = 2
ADDRESS: Final = re.compile(r" at 0x[0-9a-fA-F]+")
OBJECT: Final = re.compile(r"<([\w.]+) object")
BOUND_METHOD: Final = re.compile(r"bound method ([\w.]+)")


def _callback_type(text: str) -> str:
    stripped: Final = ADDRESS.sub("", text)
    if (instance := OBJECT.search(stripped)) is not None:
        return instance.group(1).split(".")[-1]
    if (method := BOUND_METHOD.search(stripped)) is not None:
        return method.group(1)
    return stripped.strip()


def _sample(candidate: Gateway) -> tuple[Counter[str], int]:
    response: Final = candidate.request("GET", "/active/callbacks")
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    callbacks: Final = body["all_litellm_callbacks"]
    alerting: Final = body["num_alerting"]
    assert isinstance(callbacks, list) and isinstance(alerting, int), body
    return Counter(_callback_type(str(callback)) for callback in callbacks), alerting


Samples = tuple[Counter[str], ...]


def _kinds(samples: Samples) -> frozenset[str]:
    return frozenset(chain.from_iterable(samples))


def _series(samples: Samples, kind: str) -> tuple[int, ...]:
    return tuple(sample.get(kind, 0) for sample in samples)


def _deltas(series: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(after - before for before, after in zip(series, series[1:]))


def _grows(series: tuple[int, ...]) -> bool:
    deltas: Final = _deltas(series)
    return (
        all(delta >= 0 for delta in deltas)
        and series[-1] - series[0] >= LEAK_MIN_NET_GROWTH
        and sum(1 for delta in deltas if delta > 0) >= LEAK_MIN_GROWING_INTERVALS
    )


def _grows_only_in_last_interval(series: tuple[int, ...]) -> bool:
    deltas: Final = _deltas(series)
    return all(delta >= 0 for delta in deltas) and [
        index for index, delta in enumerate(deltas) if delta > 0
    ] == [len(deltas) - 1]


def _leaking(samples: Samples) -> dict[str, tuple[int, ...]]:
    return {kind: _series(samples, kind) for kind in sorted(_kinds(samples)) if _grows(_series(samples, kind))}


def _config(directory: Path, upstream_url: str, model: str, extra: dict[str, JsonValue]) -> Path:
    config: Final = directory / f"callback_leak_{uuid.uuid4().hex}.yaml"
    general_settings: Final = extra.get("general_settings")
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_base": f"{upstream_url}/v1",
                            "api_key": "integration-provider-key",
                        },
                    }
                ],
                "router_settings": extra.get("router_settings", {}),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    **(general_settings if isinstance(general_settings, dict) else {}),
                },
            }
        )
    )
    return config


def _interval(candidate: Gateway, model: str, index: int) -> tuple[Counter[str], int]:
    for request in range(REQUESTS_PER_INTERVAL if index else 0):
        assert candidate.chat(model, text=f"leak probe {index} {request}")["model"] == model
    return _sample(candidate)


def _sample_under_traffic(candidate: Gateway, model: str) -> tuple[Samples, tuple[int, ...]]:
    taken: Final = tuple(_interval(candidate, model, index) for index in range(SAMPLES))
    callbacks: Final = tuple(sample for sample, _ in taken)
    late_growth: Final = any(_grows_only_in_last_interval(_series(callbacks, kind)) for kind in _kinds(callbacks))
    confirmed: Final = taken + ((_interval(candidate, model, SAMPLES),) if late_growth else ())
    return tuple(sample for sample, _ in confirmed), tuple(alerting for _, alerting in confirmed)


def test_callback_registry_does_not_grow_with_traffic(gateway: Gateway, tmp_path: Path) -> None:
    model: Final = f"integration-callback-leak-{uuid.uuid4().hex}"
    config: Final = _config(tmp_path, gateway.upstream_url, model, {})
    with owned_proxy(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config) as candidate:
        samples, _ = _sample_under_traffic(candidate, model)
    assert sum(samples[0].values()) > 0
    assert _leaking(samples) == {}, samples


def test_callback_registry_does_not_grow_under_latency_routing_with_alerting(gateway: Gateway, tmp_path: Path) -> None:
    model: Final = f"integration-callback-leak-{uuid.uuid4().hex}"
    config: Final = _config(
        tmp_path,
        gateway.upstream_url,
        model,
        {
            "router_settings": {"routing_strategy": "latency-based-routing"},
            "general_settings": {
                "alert_to_webhook_url": {"llm_exceptions": "http://127.0.0.1:9/integration-alerts"},
                "alert_types": ["llm_exceptions", "db_exceptions"],
            },
        },
    )
    with owned_proxy(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config) as candidate:
        samples, alerts = _sample_under_traffic(candidate, model)
    assert sum(samples[0].values()) > 0
    assert any(kind.startswith("LowestLatencyLoggingHandler") for kind in samples[0]), samples[0]
    assert _leaking(samples) == {}, samples
    assert len(set(alerts)) == 1, alerts
