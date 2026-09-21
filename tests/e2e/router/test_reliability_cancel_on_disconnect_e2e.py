"""Live e2e: a client hanging up on an in-flight request must not bench the healthy
deployment that was serving it.

The proxy runs with `cancel_on_disconnect: true`, so when the client closes the
socket before the answer arrives the proxy cancels the upstream call. That
cancellation is the client's doing, so it must never count as a failure of the
deployment: a deployment that benches on its very first failure of any kind has
to keep serving the next request, and a request served right after the hang-up
has to come from that same deployment rather than its zero-weight backup.

The disconnect is real: a non-streaming /chat/completions asking the real Azure
OpenAI deployment for a long generation, with the client closing the connection
ABANDON_AFTER_SECONDS in, well before any answer. If Azure ever answers within
that window the test fails loudly rather than passing without a disconnect.
"""

from __future__ import annotations

import time

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import AbandonedRequest
from lifecycle import ResourceManager
from models import ChatMessage, ReliabilityChatBody, RouterSettingsOverride
from reliability_support import (
    chat_override,
    create_azure_benched_on_first_failure_deployment,
    create_zero_weight_backup_deployment,
    model_id_of,
)

pytestmark = pytest.mark.e2e

ABANDON_AFTER_SECONDS = 2.0
LONG_GENERATION_MAX_TOKENS = 4000
FOLLOW_UP_CALLS = 3
FOLLOW_UP_SPACING_SECONDS = 1.0
COOLDOWN_SECONDS = 60.0


def _long_generation_prompt(marker: str) -> str:
    return (
        f"Write a detailed, multi-chapter short story of at least 3000 words about {marker}. "
        "Do not stop early and do not summarize."
    )


class TestReliabilityCancelOnDisconnect:
    @pytest.mark.covers("reliability.cooldown.client_disconnect.stays_healthy")
    def test_client_disconnect_does_not_bench_healthy_deployment(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-cancel-on-disconnect-{unique_marker()}"
        azure_deployment = create_azure_benched_on_first_failure_deployment(
            client.proxy, group, cooldown_time=COOLDOWN_SECONDS
        )
        resources.defer(lambda: client.proxy.delete_model(azure_deployment))
        backup_deployment = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup_deployment))

        abandoned = client.proxy.transport.abandon(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ReliabilityChatBody(
                model=group,
                messages=[ChatMessage(role="user", content=_long_generation_prompt(unique_marker()))],
                max_tokens=LONG_GENERATION_MAX_TOKENS,
                stream=False,
                router_settings_override=RouterSettingsOverride(num_retries=0),
                cache={"no-cache": True},
            ),
            after=ABANDON_AFTER_SECONDS,
        )
        assert isinstance(abandoned, AbandonedRequest), (
            f"the proxy answered within {ABANDON_AFTER_SECONDS}s so the client never disconnected mid-request, "
            f"got {abandoned.status_code}: {abandoned.body[:300]}"
        )

        for attempt in range(1, FOLLOW_UP_CALLS + 1):
            time.sleep(FOLLOW_UP_SPACING_SECONDS)
            resp = chat_override(
                client.proxy,
                scoped_key,
                group,
                f"say hi {unique_marker()}",
                override=RouterSettingsOverride(num_retries=0),
            )
            assert resp.status_code == 200, (
                f"follow-up {attempt}/{FOLLOW_UP_CALLS} should still land on the Azure deployment the client hung up on; "
                f"landing on the backup means the cancellation was recorded as a deployment failure and benched it, "
                f"got {resp.status_code}: {resp.body[:300]}"
            )
            assert model_id_of(resp) == azure_deployment, (
                f"follow-up {attempt}/{FOLLOW_UP_CALLS} should still land on the Azure deployment the client hung up on; "
                f"landing on the backup means the cancellation was recorded as a deployment failure and benched it, "
                f"got {model_id_of(resp)!r}"
            )
