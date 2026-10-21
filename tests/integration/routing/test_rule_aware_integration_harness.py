from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.process import owned_proxy_process
from integration._support.routing import assert_routing_matches_policy, install_rule_aware_harness, trace_path
from integration._support.wire import wire_server
from integration.providers._anthropic_safeguards_beta_support import (
    ANTHROPIC_DEPLOYMENT,
    ANTHROPIC_KEY,
    anthropic_peer,
    deliveries,
    new_marker,
    owned_config,
)

from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout

_REQUIRED_NATIVE_RULES: Final = (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED, providers=frozenset({"anthropic"})),)


def install_required_native_test_harness() -> None:
    setattr(catalog, "RULES", _REQUIRED_NATIVE_RULES)
    install_rule_aware_harness()


def _native_overrides(directory: Path) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "LITELLM_RUST": "true",
            "LITELLM_INTEGRATION_ROUTING_TRACE_DIR": str(directory / "routing-traces"),
            "LITELLM_WORKER_STARTUP_HOOKS": (
                "integration.routing.test_rule_aware_integration_harness:install_required_native_test_harness"
            ),
        }
    )


@pytest.mark.timeout(120)
def test_required_native_profile_confirms_supported_messages_request(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    with wire_server(anthropic_peer) as wire:
        config: Final = owned_config(tmp_path, gateway, wire.url)
        with owned_proxy_process(gateway, tmp_path, _native_overrides(tmp_path), config=config) as owned:
            marker: Final = new_marker()
            response: Final = owned.gateway.request(
                "POST",
                "/v1/messages",
                {
                    "model": ANTHROPIC_DEPLOYMENT,
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": marker}],
                    "cache": {"no-cache": True},
                },
            )
            assert response.status_code == 200, response.text
            assert f"msg_{marker}" in response.text, response.text
            assert response.headers.get("x-litellm-rust") == "true", dict(response.headers)
            trace_id: Final = response.headers["x-integration-routing-id"]
            assert_routing_matches_policy(response.headers, trace_path(tmp_path / "routing-traces", trace_id))
        sent: Final = deliveries(wire)

    assert len(sent) == 1, sent
    assert sent[0].api_key == ANTHROPIC_KEY, sent
    assert sent[0].target == "/v1/messages", sent
    assert sent[0].marker == marker, sent
