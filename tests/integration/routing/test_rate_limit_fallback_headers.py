from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import OwnedState, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import wire_server
from integration.routing.test_per_model_rate_limit_fallbacks import (
    assert_served,
    chat,
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
    second_fallback: str


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    suffix: Final = uuid.uuid4().hex[:10]
    primary: Final = f"quota-primary-{suffix}"
    fallback: Final = f"quota-fallback-{suffix}"
    second_fallback: Final = f"quota-second-fallback-{suffix}"
    directory: Final = tmp_path_factory.mktemp("rate-limit-fallback-headers")
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        primary_wire: Final = stack.enter_context(wire_server(healthy_reply))
        fallback_wire: Final = stack.enter_context(wire_server(healthy_reply))
        second_fallback_wire: Final = stack.enter_context(wire_server(healthy_reply))
        config: Final = proxy_config(
            directory,
            model_list=(
                model_entry(primary, primary_wire),
                model_entry(fallback, fallback_wire),
                model_entry(second_fallback, second_fallback_wire),
            ),
            fallbacks=({primary: [fallback, second_fallback]},),
            general_settings={},
        )
        owned: Final = stack.enter_context(
            owned_proxy_process(
                rig_gateway, directory, {}, config=config, workers=2, state=OwnedState.SHARED_WITH_WORKER
            )
        )
        yield Rig(owned.gateway, primary, fallback, second_fallback)


def test_quota_fallback_response_names_the_hop_and_its_reason(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_tpm_limit={rig.primary: 1})

        response: Final = chat(rig.gateway, key, rig.primary, f"pmrl-{uuid.uuid4().hex}")
        assert_served(response, rig.fallback, "served pmrl-")
        assert response.headers.get("x-litellm-attempted-fallbacks") == "1", dict(response.headers)
        assert response.headers.get("x-litellm-fallback-reason") == "rate_limit:model_per_key", dict(response.headers)


def test_quota_fallback_response_counts_every_candidate_a_limit_rejected(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_tpm_limit={rig.primary: 1, rig.fallback: 1})

        response: Final = chat(rig.gateway, key, rig.primary, f"pmrl-{uuid.uuid4().hex}")
        assert_served(response, rig.second_fallback, "served pmrl-")
        assert response.headers.get("x-litellm-attempted-fallbacks") == "2", dict(response.headers)
        assert response.headers.get("x-litellm-fallback-reason") == "rate_limit:model_per_key", dict(response.headers)
