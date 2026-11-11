from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import OwnedState, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import wire_server
from integration.routing.test_fallback_rate_limit_check import provider_error_reply
from integration.routing.test_per_model_rate_limit_fallbacks import (
    assert_served,
    chat,
    healthy_reply,
    model_entry,
    proxy_config,
)
from redis import Redis

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

ADMISSION_PROBE_TOKENS: Final = 1
SETTLEMENT_DEADLINE_SECONDS: Final = 15.0


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    primary: str
    fallback: str


def key_model_tokens(group: str) -> int:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), socket_timeout=2) as cache:
        counters: Final = tuple(cache.scan_iter(match=f"*model_per_key:*:{group}}}:tokens"))
        return sum(int(cache.get(counter) or 0) for counter in counters)


def settled_key_model_tokens(primary: str, fallback: str, total_tokens: int) -> tuple[int, int]:
    deadline: Final = time.monotonic() + SETTLEMENT_DEADLINE_SECONDS
    while True:
        charged: Final = (key_model_tokens(primary), key_model_tokens(fallback))
        if (charged[0] == 0 and charged[1] >= total_tokens) or time.monotonic() >= deadline:
            return charged
        time.sleep(0.5)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    suffix: Final = uuid.uuid4().hex[:10]
    primary: Final = f"settle-primary-{suffix}"
    fallback: Final = f"settle-fallback-{suffix}"
    directory: Final = tmp_path_factory.mktemp("fallback-settles-serving-model")
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
        owned: Final = stack.enter_context(
            owned_proxy_process(
                rig_gateway, directory, {}, config=config, workers=2, state=OwnedState.SHARED_WITH_WORKER
            )
        )
        yield Rig(owned.gateway, primary, fallback)


def test_provider_error_fallback_charges_its_tokens_to_the_model_that_served(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_tpm_limit={rig.primary: 100_000, rig.fallback: 1_000})

        response: Final = chat(rig.gateway, key, rig.primary, f"pmrl-{uuid.uuid4().hex}")
        assert_served(response, rig.fallback, "served pmrl-")
        total_tokens: Final = int(response.json()["usage"]["total_tokens"])
        assert total_tokens > 0, response.text

        primary_tokens, fallback_tokens = settled_key_model_tokens(rig.primary, rig.fallback, total_tokens)
        assert primary_tokens == 0, (primary_tokens, fallback_tokens)
        assert total_tokens <= fallback_tokens <= total_tokens + ADMISSION_PROBE_TOKENS, (
            primary_tokens,
            fallback_tokens,
        )
