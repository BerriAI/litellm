"""Live e2e: team metadata disable_global_guardrails opts out of default-on
guardrails, while keys not on such a team stay subject to them.

Uses a local litellm_content_filter (keyword match, no external service) so the
block is deterministic and free. Restored on ProxyClient after the Gateway-era
suite was removed.
"""

from __future__ import annotations

import time

import pytest

from e2e_config import unique_marker
from e2e_http import UnknownApiError, unwrap
from guardrails_client import GuardrailsClient
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL = "gemini-2.5-flash"

# A guardrail created via POST /guardrails is registered in-process immediately
# on the worker that served the create call, but the proxy runs multiple
# pods/workers behind the shared key, and every other one only picks up the new
# guardrail on its next periodic DB sync (every 30s), so the very next request
# can race a worker that has not synced yet.
GUARDRAIL_PROPAGATION_DEADLINE_SECONDS = 40.0
GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS = 5.0


def _prompt_with(banned_keyword: str) -> str:
    return f"Reply with the single word OK. {banned_keyword}"


def _assert_eventually_blocked(client: GuardrailsClient, key: str, banned: str) -> None:
    deadline = time.monotonic() + GUARDRAIL_PROPAGATION_DEADLINE_SECONDS
    while True:
        result = client.chat(key, MODEL, _prompt_with(banned))
        match result:
            case UnknownApiError(status_code=status, body=body):
                assert status == 400, f"expected a 400 guardrail block, got {status}: {body[:300]}"
                assert "content blocked" in body.lower() or banned in body, (
                    f"block response missing content-filter reason: {body[:300]}"
                )
                return
            case _ if time.monotonic() < deadline:
                time.sleep(GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS)
            case _:
                pytest.fail(
                    f"default-on guardrail never blocked the banned keyword within "
                    f"{GUARDRAIL_PROPAGATION_DEADLINE_SECONDS}s; got {result}"
                )


class TestTeamDisableGlobalGuardrail:
    @pytest.mark.covers(
        "guardrail.litellm_content_filter.pre_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_global_guardrail_blocks_key_without_team_opt_out(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        banned = unique_marker()
        guardrail_id = client.create_content_filter_guardrail(f"e2e-content-filter-{banned}", banned)
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        _assert_eventually_blocked(client, scoped_key, banned)

    @pytest.mark.covers(
        "guardrail.litellm_content_filter.pre_call.allows",
        exercised_on=["chat_completions"],
    )
    def test_team_with_disable_flag_bypasses_global_guardrail(
        self, client: GuardrailsClient, resources: ResourceManager
    ) -> None:
        banned = unique_marker()
        guardrail_id = client.create_content_filter_guardrail(f"e2e-content-filter-{banned}", banned)
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        team_id = client.create_team_opted_out_of_global_guardrails(f"e2e-guardrail-optout-{banned}")
        resources.defer(lambda: client.delete_team(team_id))
        key = client.create_key_in_team(team_id)
        resources.defer(lambda: client.proxy.delete_key(key))

        chat = unwrap(client.chat(key, MODEL, _prompt_with(banned)))

        assert chat.choices, (
            f"team opted out of global guardrails, so the banned keyword must pass "
            f"through and the call must succeed, but no choices came back: {chat}"
        )
