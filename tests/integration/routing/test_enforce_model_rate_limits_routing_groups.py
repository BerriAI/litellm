from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import gateway_from_environment
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import owned_config
from integration.routing.test_payment_required_mapping import OPENAI_MODEL, is_model_info_probe, model_list_reply
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)

RPM: Final = 10
EXTRA_REQUESTS: Final = 2
LATENCY_GROUP: Final = "lat-model"
USAGE_GROUP: Final = "ubr-model"


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    latency: Wire
    usage: Wire


def _reply(request: Request) -> Reply:
    if is_model_info_probe(request):
        return model_list_reply()
    return chat_reply("chatcmpl-rpm", OPENAI_MODEL, "served", stream=False)


def _deployment(model_name: str, model_id: str, wire: Wire) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": OPENAI_MODEL,
            "api_base": wire.url + "/v1",
            "api_key": "synthetic-openai-key",
            "rpm": RPM,
        },
        "model_info": {"id": model_id},
    }


def _config(directory: Path, latency: Wire, usage: Wire) -> Path:
    """``enforce_model_rate_limits`` next to a usage-based-routing-v2 routing group, as the customer ran it."""
    return owned_config(
        directory,
        [_deployment(LATENCY_GROUP, "lat-a", latency), _deployment(USAGE_GROUP, "ubr-a", usage)],
        litellm_settings={"cache": False},
        router_settings={
            "optional_pre_call_checks": ["enforce_model_rate_limits"],
            "num_retries": 0,
            "routing_groups": [
                {"group_name": "latency-group", "models": [LATENCY_GROUP], "routing_strategy": "latency-based-routing"},
                {"group_name": "ubr-group", "models": [USAGE_GROUP], "routing_strategy": "usage-based-routing-v2"},
            ],
        },
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("rpm-routing-groups")
    with (
        gateway_from_environment() as gateway,
        wire_server(_reply) as latency,
        wire_server(_reply) as usage,
        owned_proxy_process(gateway, directory, {}, config=_config(directory, latency, usage)) as owned,
    ):
        yield Rig(owned, latency, usage)


def _chat(rig: Rig, model: str) -> httpx.Response:
    return rig.owned.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"rpm {uuid.uuid4().hex}"}]},
    )


def _forwarded(wire: Wire) -> int:
    return sum(1 for request in wire.drain() if not is_model_info_probe(request))


@pytest.mark.parametrize(
    ("model", "wire"), [(LATENCY_GROUP, "latency"), (USAGE_GROUP, "usage")], ids=["other-group", "v2-group"]
)
def test_a_deployment_with_rpm_10_serves_ten_requests_a_minute(rig: Rig, model: str, wire: str) -> None:
    responses: Final = tuple(_chat(rig, model) for _ in range(RPM + EXTRA_REQUESTS))

    assert [response.status_code for response in responses] == [200] * RPM + [429] * EXTRA_REQUESTS, [
        response.text for response in responses
    ]
    assert _forwarded(getattr(rig, wire)) == RPM
