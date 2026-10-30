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

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

GROUP: Final = "daily-limited-group"
LIMITED_ID: Final = "free-tier-two-a-day"
UNLIMITED_ID: Final = "paid-tier"
RPD: Final = 2
EXTRA_REQUESTS: Final = 2


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    limited: Wire
    unlimited: Wire


def _reply(request: Request) -> Reply:
    if is_model_info_probe(request):
        return model_list_reply()
    return chat_reply("chatcmpl-daily", OPENAI_MODEL, "served", stream=False)


def _deployment(model_id: str, wire: Wire, **params: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": GROUP,
        "litellm_params": {
            "model": OPENAI_MODEL,
            "api_base": wire.url + "/v1",
            "api_key": "synthetic-openai-key",
            **params,
        },
        "model_info": {"id": model_id},
    }


def _config(directory: Path, limited: Wire, unlimited: Wire) -> Path:
    return owned_config(
        directory,
        [_deployment(LIMITED_ID, limited, rpd=RPD, order=1), _deployment(UNLIMITED_ID, unlimited, order=2)],
        litellm_settings={"cache": False},
        router_settings={"optional_pre_call_checks": ["enforce_model_rate_limits"], "num_retries": 1},
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("daily-limits")
    with (
        gateway_from_environment() as gateway,
        wire_server(_reply) as limited,
        wire_server(_reply) as unlimited,
        owned_proxy_process(gateway, directory, {}, config=_config(directory, limited, unlimited)) as owned,
    ):
        yield Rig(owned, limited, unlimited)


def _chat(rig: Rig) -> httpx.Response:
    return rig.owned.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": GROUP, "messages": [{"role": "user", "content": f"daily {uuid.uuid4().hex}"}]},
    )


def _forwarded(wire: Wire) -> int:
    return sum(1 for request in wire.drain() if not is_model_info_probe(request))


def test_requests_past_the_deployment_rpd_are_served_by_the_other_deployment_in_the_group(rig: Rig) -> None:
    responses: Final = tuple(_chat(rig) for _ in range(RPD + EXTRA_REQUESTS))

    assert [response.status_code for response in responses] == [200] * (RPD + EXTRA_REQUESTS), [
        response.text for response in responses
    ]
    assert [response.headers["x-litellm-model-id"] for response in responses] == [LIMITED_ID] * RPD + [
        UNLIMITED_ID
    ] * EXTRA_REQUESTS
    assert (_forwarded(rig.limited), _forwarded(rig.unlimited)) == (RPD, EXTRA_REQUESTS)
