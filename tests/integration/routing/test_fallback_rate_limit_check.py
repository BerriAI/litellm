from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.routing.test_payment_required_mapping import is_model_info_probe, model_list_reply
from integration.routing.test_per_model_rate_limit_fallbacks import (
    assert_served,
    chat,
    error_message,
    healthy_reply,
    model_entry,
    proxy_config,
)

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    primary: str
    fallback: str
    fallback_wire: Wire


def provider_error_reply(request: Request) -> Reply:
    if is_model_info_probe(request):
        return model_list_reply()
    return Reply(status=500, body=b'{"error": {"message": "scripted provider outage", "type": "server_error"}}')


def served_requests(wire: Wire) -> int:
    return sum(1 for request in wire.drain() if not is_model_info_probe(request))


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    suffix: Final = uuid.uuid4().hex[:10]
    primary: Final = f"fallback-check-primary-{suffix}"
    fallback: Final = f"fallback-check-fallback-{suffix}"
    directory: Final = tmp_path_factory.mktemp("fallback-rate-limit-check")
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        primary_wire: Final = stack.enter_context(wire_server(provider_error_reply))
        fallback_wire: Final = stack.enter_context(wire_server(healthy_reply))
        config: Final = proxy_config(
            directory,
            model_list=(model_entry(primary, primary_wire), model_entry(fallback, fallback_wire)),
            fallbacks=({primary: [fallback]},),
            general_settings={},
        )
        owned: Final = stack.enter_context(owned_proxy_process(rig_gateway, directory, {}, config=config, workers=2))
        yield Rig(owned.gateway, primary, fallback, fallback_wire)


def test_provider_error_fallback_honors_the_keys_per_model_rpm_limit_on_the_fallback_model(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={rig.fallback: 1})
        served_requests(rig.fallback_wire)

        first: Final = chat(rig.gateway, key, rig.primary, f"pmrl-{uuid.uuid4().hex}")
        assert_served(first, rig.fallback, "served pmrl-")

        second: Final = chat(rig.gateway, key, rig.primary, f"pmrl-{uuid.uuid4().hex}")
        assert second.status_code == 500, (second.status_code, second.text)
        assert f"Fallback to {rig.fallback} was skipped because the caller is over its per-model rate limit" in (
            error_message(second.text)
        ), second.text
        assert served_requests(rig.fallback_wire) == 1
