"""Live e2e: team-scoped Langfuse callback delivery and isolation.

Covers logging.langfuse.success.logs_spend: a team configured with a Langfuse
callback via POST /team/{id}/callback must deliver its members' calls to the
real Langfuse project (generation readable back through Langfuse's own API,
with the cost agreeing with the x-litellm-response-cost header), while traffic
from keys outside the team must NOT reach that project - the isolation is the
point of team-scoped callbacks.

Both halves of the contract are asserted: the recorded state (the /team/callback
registration itself answers success) and the enforced behavior (the generation
at the destination for the team key, and its absence for the non-team key).
"""

from __future__ import annotations

import time

import pytest

from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from lifecycle import ResourceManager
from logging_client import (
    LangfuseCreds,
    LoggingClient,
    costs_agree,
    first_ok,
    load_langfuse_creds,
    observation_spend,
)

pytestmark = pytest.mark.e2e

#: How long to keep re-checking that the non-team call never surfaces in
#: Langfuse after the team call's generation has already been ingested; the
#: positive observation bounds the pipeline's latency, so a wrong delivery
#: would be visible within the same order of magnitude.
ISOLATION_SETTLE_SECONDS = 30.0
ISOLATION_CHECK_INTERVAL_SECONDS = 5.0


@pytest.fixture(scope="session")
def langfuse_creds() -> LangfuseCreds:
    return load_langfuse_creds()


class TestTeamLangfuseCallback:
    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["chat_completions"])
    def test_team_callback_delivers_and_isolates(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        team_id = client.create_team(f"lf-team-{unique_marker()}", models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_team(team_id))
        # Recorded state: the registration endpoint itself must answer success
        # (add_team_langfuse_callback asserts it).
        client.add_team_langfuse_callback(team_id, langfuse_creds)

        team_alias = f"lf-team-key-{unique_marker()}"
        team_key = client.key_with_alias(team_alias, models=[CHEAP_ANTHROPIC_MODEL], team_id=team_id)
        resources.defer(lambda: client.delete_key(team_key))
        solo_alias = f"lf-solo-key-{unique_marker()}"
        solo_key = client.key_with_alias(solo_alias, models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(solo_key))

        # Enforced behavior, positive half, with one propagation retry: a
        # worker still holding the pre-callback team object can serve the
        # first call without shipping it, and by the time the first Langfuse
        # poll has timed out the team cache TTL has lapsed, so a second call
        # must deliver.
        team_marker = ""
        team_outcome = None
        observation = None
        for _attempt in range(2):
            team_marker = unique_marker()
            team_outcome = first_ok(
                client,
                lambda marker=team_marker: client.chat_raw(
                    team_key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", max_tokens=16
                ),
            )
            assert team_outcome.response_cost is not None and team_outcome.response_cost > 0, (
                f"the response must report x-litellm-response-cost, got {team_outcome.response_cost!r}"
            )
            observation = client.poll_langfuse_observation(
                langfuse_creds,
                key_alias=team_alias,
                prompt_marker=team_marker,
                require_positive_cost=True,
            )
            if observation is not None:
                break
        solo_marker = unique_marker()
        _ = first_ok(
            client,
            lambda: client.chat_raw(
                solo_key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {solo_marker}", max_tokens=16
            ),
        )

        assert observation is not None, (
            f"the team key's call (marker {team_marker}) never reached Langfuse within the deadline, "
            "even after a fresh call past the team-object cache TTL"
        )
        assert team_outcome is not None and team_outcome.response_cost is not None
        cost = observation_spend(observation)
        assert cost is not None and costs_agree(team_outcome.response_cost, cost), (
            f"Langfuse calculatedTotalCost {cost!r} must agree with the header cost {team_outcome.response_cost}"
        )

        # Enforced behavior, negative half: the non-team call must never show
        # up in this project. The positive generation above has already been
        # ingested, which bounds the pipeline latency, so keep re-checking for
        # a settle window rather than trusting a single instant.
        settle_deadline = time.monotonic() + ISOLATION_SETTLE_SECONDS
        while True:
            leaked = client.find_langfuse_observation(langfuse_creds, key_alias=solo_alias, prompt_marker=solo_marker)
            assert leaked is None, (
                f"a non-team key's call (marker {solo_marker}) reached the team's Langfuse "
                f"project: {leaked.id} - team callbacks must not apply outside the team"
            )
            if time.monotonic() >= settle_deadline:
                break
            time.sleep(ISOLATION_CHECK_INTERVAL_SECONDS)
