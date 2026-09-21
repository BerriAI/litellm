"""Live e2e: a client hanging up mid-request under cancel_on_disconnect never
benches the deployment it was talking to.

With `general_settings.cancel_on_disconnect: true` the proxy cancels the in-flight
provider call the moment the client's socket closes. The Azure handler used to
turn that cancellation into a fake 500, which the router booked as a deployment
failure: one impatient client benched a healthy deployment and every caller
behind it paid for fallbacks (GitHub issues #35329 and #42222). This cell pins the
fix at the seam a customer sees. The group is the cooldown suite's pair: the live
Azure deployment holding all of the shuffle weight, benched on its first failure
of any class (the fake 500 carried no provider body, so litellm mapped it to a
bare APIError no named policy class covers) with a cooldown long enough to
outlast the test, plus a healthy backup at weight 0 the shuffle can only reach
once the Azure deployment is benched. One cheap call first proves the Azure
deployment answers the key and leaves the key's auth path warm. The test then
asks for a long answer, retries off, and hangs up a few seconds in: the client's
read timeout closes the socket well after the proxy has handed the call to Azure
(a cold virtual-key auth can take a couple of seconds on its own, and a hang-up
that lands before the provider call is in flight cancels nothing the router could
bench, so a shorter window passes vacuously) and well before the answer is done.
After a settle window wide enough for a sibling replica to have read any bench
from Redis, every one of the next calls has to come back 200 from the Azure
deployment itself, named in x-litellm-model-id; a single answer from the backup
means the hang-up was booked as a failure.

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
    chat_override,
    create_azure_benched_on_first_failure_deployment,
    create_zero_weight_backup_deployment,
    model_id_of,
)

pytestmark = pytest.mark.e2e

CLIENT_HANGS_UP_AFTER_SECONDS = 8.0
LONG_ANSWER_MAX_TOKENS = 4096
BENCH_OUTLASTS_TEST_SECONDS = 300.0
SETTLE_AFTER_HANGUP_SECONDS = 3.0
CALLS_AFTER_HANGUP = 6


def _say_hi(client: ComplexityRouterClient, key: str, group: str) -> StreamingResponse:
    return chat_override(
        client.proxy,
        key,
        group,
        f"say hi {unique_marker()}",
        override=RouterSettingsOverride(num_retries=0),
    )


def _hang_up_mid_answer(client: ComplexityRouterClient, key: str, group: str) -> None:
    """Send a request whose answer takes far longer than the client waits, so the
    client closes the socket while the provider is still generating."""
    outcome = client.proxy.transport.abandon(
        "/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=ReliabilityChatBody(
            model=group,
            messages=[
                ChatMessage(
                    role="user",
                    content=f"Write a 3000 word essay on the history of the telegraph. {unique_marker()}",
                )
            ],
            max_tokens=LONG_ANSWER_MAX_TOKENS,
            router_settings_override=RouterSettingsOverride(num_retries=0),
        ),
        after=CLIENT_HANGS_UP_AFTER_SECONDS,
    )
    match outcome:
        case AbandonedRequest():
            return
        case StreamingResponse(status_code=status_code, body=body):
            pytest.fail(
                f"the client should have hung up {CLIENT_HANGS_UP_AFTER_SECONDS:.0f}s into a long answer with the "
                f"call still in flight, but the proxy answered first with {status_code}: {body[:300]}"
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
        time.sleep(SETTLE_AFTER_HANGUP_SECONDS)

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
