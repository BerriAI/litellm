"""Live e2e: a client hanging up mid-request under cancel_on_disconnect never
benches the deployment it was talking to.

The group is the cooldown suite's pair: the live Azure deployment holding all of
the shuffle weight, benched on its first failure of any class with a cooldown that
outlasts the test, plus a healthy backup at weight 0 the shuffle only reaches once
the Azure deployment is benched. A cheap call first proves the Azure deployment
answers the key and warms its auth path. The test then asks for an answer far
longer than CLIENT_HANGS_UP_AFTER_SECONDS of generation, retries off, and hangs up
that many seconds in: late enough that the proxy has handed the call to Azure (a
hang-up before the provider call is in flight cancels nothing the router could
bench, so the cell would pass vacuously). An answer that comes back inside the
window proves nothing and benches nothing either, since a success never counts
against the deployment, so the cell asks again up to HANG_UP_ATTEMPTS times and
fails out loud naming the window only when every ask came back early. After the
cooldown suite's replica propagation window, every one of the next calls has to
come back 200 from the Azure deployment itself, named in x-litellm-model-id; a
single answer from the backup means the hang-up was booked as a failure.

The test reads `cancel_on_disconnect` back from the proxy first: without the flag
the hang-up cancels nothing and the cell would pass vacuously.
"""

from __future__ import annotations

import time

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import AbandonedRequest, StreamingResponse
from lifecycle import ResourceManager
from models import ChatMessage, ReliabilityChatBody, RouterSettingsOverride
from reliability_support import (
    REPLICA_PROPAGATION_SECONDS,
    chat_override,
    create_azure_benched_on_first_failure_deployment,
    create_zero_weight_backup_deployment,
    model_id_of,
)

pytestmark = pytest.mark.e2e

CLIENT_HANGS_UP_AFTER_SECONDS = 5.0
HANG_UP_ATTEMPTS = 3
LONG_ANSWER_MAX_TOKENS = 16384
BENCH_OUTLASTS_TEST_SECONDS = 300.0
CALLS_AFTER_HANGUP = 6


def _say_hi(client: ComplexityRouterClient, key: str, group: str) -> StreamingResponse:
    return chat_override(
        client.proxy,
        key,
        group,
        f"say hi {unique_marker()}",
        override=RouterSettingsOverride(num_retries=0),
    )


def _ask_for_a_long_answer_then_hang_up(
    client: ComplexityRouterClient, key: str, group: str
) -> AbandonedRequest | StreamingResponse:
    return client.proxy.transport.abandon(
        "/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=ReliabilityChatBody(
            model=group,
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Write an essay on the history of the telegraph with one section per decade from the 1830s "
                        f"to the 2020s, each section at least 300 words. {unique_marker()}"
                    ),
                )
            ],
            max_tokens=LONG_ANSWER_MAX_TOKENS,
            router_settings_override=RouterSettingsOverride(num_retries=0),
        ),
        after=CLIENT_HANGS_UP_AFTER_SECONDS,
    )


def _hang_up_mid_answer(client: ComplexityRouterClient, key: str, group: str) -> None:
    for attempt in range(1, HANG_UP_ATTEMPTS + 1):
        match _ask_for_a_long_answer_then_hang_up(client, key, group):
            case AbandonedRequest():
                return
            case StreamingResponse(status_code=200):
                continue
            case StreamingResponse(status_code=status_code, body=body):
                pytest.fail(
                    f"hang-up attempt {attempt} should have found the long answer still in flight after "
                    f"{CLIENT_HANGS_UP_AFTER_SECONDS:.0f}s, but the proxy answered {status_code}: {body[:300]}"
                )
    pytest.fail(
        f"the proxy answered all {HANG_UP_ATTEMPTS} long asks within {CLIENT_HANGS_UP_AFTER_SECONDS:.0f}s, so the "
        "client never hung up with a call still in flight and the bench this cell guards against could not happen"
    )


class TestReliabilityCancelOnDisconnect:
    @pytest.mark.covers("reliability.cooldown.client_disconnect.stays_healthy")
    def test_client_hanging_up_never_benches_the_deployment(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        assert client.proxy.general_setting_enabled("cancel_on_disconnect"), (
            "this cell needs general_settings.cancel_on_disconnect: true in the proxy config; without it the "
            "hang-up cancels nothing and the bench it guards against can never happen"
        )

        group = f"reliability-cooldown-disconnect-{unique_marker()}"
        azure = create_azure_benched_on_first_failure_deployment(
            client.proxy, group, cooldown_time=BENCH_OUTLASTS_TEST_SECONDS
        )
        resources.defer(lambda: client.proxy.delete_model(azure))
        backup = create_zero_weight_backup_deployment(client.proxy, group)
        resources.defer(lambda: client.proxy.delete_model(backup))

        warm_up = _say_hi(client, scoped_key, group)
        assert warm_up.status_code == 200 and model_id_of(warm_up) == azure, (
            f"before any hang-up the Azure deployment {azure} should answer the group, got {warm_up.status_code} "
            f"from {model_id_of(warm_up)!r}: {warm_up.body[:300]}"
        )

        _hang_up_mid_answer(client, scoped_key, group)
        time.sleep(REPLICA_PROPAGATION_SECONDS)

        for call in range(1, CALLS_AFTER_HANGUP + 1):
            resp = _say_hi(client, scoped_key, group)
            assert resp.status_code == 200, (
                f"call {call} after the hang-up should have been a plain 200 from the group, got "
                f"{resp.status_code}: {resp.body[:300]}"
            )
            assert model_id_of(resp) == azure, (
                f"call {call} after the hang-up should have been served by the Azure deployment {azure}, the proxy "
                f"named {model_id_of(resp)!r}: the cancelled call was booked as a failure and benched it"
            )
