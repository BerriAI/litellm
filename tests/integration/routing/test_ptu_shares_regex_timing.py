import statistics
import threading
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment, string_value
from integration._support.openai_wire import chat_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.timing import assert_within_factor_of_control, timed
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 240), pytest.mark.shares_worker_state]

_WORST_GROUP: Final = "ptu-regex-worst"
_CONTROL_GROUP: Final = "ptu-regex-unshared-worst"
_UPSTREAM_URL_PLACEHOLDER: Final = "upstream-url-placeholder"
_WORST_BASE_MODEL: Final = "azure/" + "-0000-00-00" * 5_958 + "x"
_ATTRIBUTION_ON: Final = MappingProxyType({"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True"})
_REQUESTS: Final = 8
_CONFIG: Final = TypeAdapter(dict[str, JsonValue])
_FACTOR: Final = 3.0


def _reply(_request: Request) -> Reply:
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", "gpt-4.1", "served", stream=False)


def _share_terms(holder: str) -> dict[str, JsonValue]:
    return {
        "ptu_count": 1,
        "cost_per_ptu_per_hour": 2.0,
        "ptu_effective_from": "2026-01-01T00:00:00Z",
        "ptu_shares": {holder: 1},
    }


def _deployment(model_name: str, model_info: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"azure/{model_name}",
            "api_base": _UPSTREAM_URL_PLACEHOLDER,
            "api_key": "synthetic-azure-key",
            "api_version": "2025-03-01-preview",
        },
        "model_info": {"id": f"{model_name}-{uuid.uuid4().hex}", "base_model": _WORST_BASE_MODEL, **model_info},
    }


def _config(directory: Path, wire_url: str, holder: str) -> Path:
    config: Final = {
        **_CONFIG.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())),
        "model_list": [
            _deployment(_CONTROL_GROUP, {}),
            _deployment(_WORST_GROUP, _share_terms(holder)),
        ],
        "router_settings": {"disable_cooldowns": True, "num_retries": 0, "ignore_invalid_deployments": True},
    }
    path: Final = directory / "ptu-regex.yaml"
    path.write_text(yaml.safe_dump(config, width=2_000_000).replace(_UPSTREAM_URL_PLACEHOLDER, wire_url))
    return path


@dataclass(frozen=True, slots=True)
class _Timing:
    proxy: Gateway
    key: str


@pytest.fixture(scope="module")
def timing(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Timing]:
    directory: Final = tmp_path_factory.mktemp("ptu-regex")
    holder: Final = f"ptu-regex-holder-{uuid.uuid4().hex}"
    with gateway_from_environment() as gateway, wire_server(_reply) as wire:
        try:
            with owned_proxy_process(
                gateway, directory, _ATTRIBUTION_ON, config=_config(directory, wire.url, holder), workers=2
            ) as owned:
                owned.gateway.post("/team/new", {"team_id": holder, "team_alias": holder})
                key: Final = string_value(owned.gateway.post("/key/generate", {"team_id": holder})["key"])
                yield _Timing(owned.gateway, key)
        finally:
            gateway.post("/team/delete", {"team_ids": [holder]})


def _chat_seconds(proxy: Gateway, key: str, model: str) -> float:
    result: Final = timed(
        lambda: proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": uuid.uuid4().hex}]},
            key=key,
        )
    )
    assert result.value.status_code == 200, result.value.text
    return result.seconds


def _liveliness_seconds(base_url: str, stop: threading.Event) -> tuple[float, ...]:
    def probe() -> float:
        with httpx.Client(
            base_url=base_url, timeout=30, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)
        ) as client:
            result: Final = timed(lambda: client.get("/health/liveliness"))
        assert result.value.status_code == 200, result.value.text
        return result.seconds

    return tuple(probe() for _ in iter(stop.is_set, True))


def _under_load(timing: _Timing, model: str) -> tuple[float, float]:
    stop: Final = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        liveliness: Final = pool.submit(_liveliness_seconds, str(timing.proxy.client.base_url), stop)
        try:
            chats: Final = tuple(_chat_seconds(timing.proxy, timing.key, model) for _ in range(_REQUESTS))
        finally:
            stop.set()
        probes: Final = liveliness.result()
    assert probes, "no liveliness probe completed while the requests ran"
    return statistics.median(chats), statistics.median(probes)


def test_a_64kb_base_model_on_a_shared_deployment_keeps_requests_and_liveliness_near_the_unshared_one(
    timing: _Timing,
) -> None:
    warmed: Final = tuple(_chat_seconds(timing.proxy, timing.key, model) for model in (_CONTROL_GROUP, _WORST_GROUP))
    assert len(warmed) == 2

    control_chat, control_liveliness = _under_load(timing, _CONTROL_GROUP)
    worst_chat, worst_liveliness = _under_load(timing, _WORST_GROUP)

    assert_within_factor_of_control(worst_chat, control=control_chat, factor=_FACTOR)
    assert_within_factor_of_control(worst_liveliness, control=control_liveliness, factor=_FACTOR)
