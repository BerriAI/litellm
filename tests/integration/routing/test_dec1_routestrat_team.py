"""Team-based routing: a deployment with `model_info.team_id` serves only that team's callers.

The first filter of selection is `filter_team_based` (`litellm/rust/crates/router-core/src/select.rs:203`): a
deployment owned by a team (`model_info.team_id`) is usable only by a request whose team is that team
(`Deployment::usable_by_team`, `registry.rs:262`), while a deployment without a team serves everyone. The one
exception is a proxy admin without a team asking for a name whose deployments are all team owned: the admin may use
them when they belong to one team, and gets HTTP 400 `model name ... matches deployments from multiple teams`
(`Error::MultipleTeams`, `error.rs:77`) when they belong to several. The strategy then picks among what is left, so
a team member can still be load balanced across its own and public deployments.

Team ids are generated per run and written into the config before the proxy boots; each test then creates the team
with that id and a key in it. The public deployment is OpenAI; team deployments are on Azure, Bedrock and Vertex.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration.routing.dec1_routestrat_support import (
    Rig,
    chat,
    dep,
    error_message,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

TEAM_A: Final = f"dec1-routestrat-team-a-{uuid.uuid4().hex[:8]}"
TEAM_B: Final = f"dec1-routestrat-team-b-{uuid.uuid4().hex[:8]}"


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep("openai", "team-mixed", "tm-public-openai", base),
        dep("azure", "team-mixed", "tm-a-azure", base, weight=1, model_info={"team_id": TEAM_A}),
        dep("bedrock", "team-only-a", "toa-bedrock", base, model_info={"team_id": TEAM_A}),
        dep("vertex", "team-split", "tsp-a-vertex", base, model_info={"team_id": TEAM_A}),
        dep("bedrock", "team-split", "tsp-b-bedrock", base, model_info={"team_id": TEAM_B}),
        dep("openai", "team-shared", "tsh-public-openai", base, weight=1),
        dep("vertex", "team-shared", "tsh-a-vertex", base, weight=1, model_info={"team_id": TEAM_A}),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "simple-shuffle",
        "num_retries": 0,
        "disable_cooldowns": True,
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-team"), _models, settings) as value:
        yield value


def test_a_team_key_reaches_its_teams_weighted_deployment_in_a_mixed_group(rig: Rig) -> None:
    """`team-mixed` has a public OpenAI deployment without weight and an Azure deployment owned by team A with
    `weight: 1`. A key of team A sends twenty requests.

    All twenty must reach the team's Azure deployment: both are usable by the team, and the weighted one wins the
    shuffle.
    """
    with rig.gateway.scenario() as scenario:
        scenario.team(team_id=TEAM_A)
        key: Final = scenario.key(team_id=TEAM_A)
        assert served_many(rig.gateway, "team-mixed", 20, key=key) == {"tm-a-azure": 20}


def test_another_teams_key_never_reaches_the_team_owned_deployment(rig: Rig) -> None:
    """A key of team B sends twenty requests to `team-mixed`.

    All twenty must reach the public OpenAI deployment, although the shuffle weight favours the Azure deployment
    owned by team A: the team filter runs before the strategy.
    """
    with rig.gateway.scenario() as scenario:
        scenario.team(team_id=TEAM_B)
        key: Final = scenario.key(team_id=TEAM_B)
        assert served_many(rig.gateway, "team-mixed", 20, key=key) == {"tm-public-openai": 20}


def test_a_key_without_a_team_only_reaches_public_deployments(rig: Rig) -> None:
    """A key with no team sends twenty requests to `team-mixed`.

    All twenty must reach the public OpenAI deployment.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        assert served_many(rig.gateway, "team-mixed", 20, key=key) == {"tm-public-openai": 20}


def test_the_master_key_without_a_team_only_reaches_public_deployments_of_a_mixed_group(rig: Rig) -> None:
    """The proxy admin (master key, no team) sends twenty requests to `team-mixed`.

    All twenty must reach the public OpenAI deployment: the admin exception applies only when every deployment of
    the name is team owned.
    """
    assert served_many(rig.gateway, "team-mixed", 20) == {"tm-public-openai": 20}


def test_the_master_key_may_call_a_name_served_only_by_one_teams_deployment(rig: Rig) -> None:
    """`team-only-a` has a single Bedrock deployment owned by team A. The master key sends five requests.

    All five must be served by that deployment (`select.rs:208`, the proxy-admin exception).
    """
    assert served_many(rig.gateway, "team-only-a", 5) == {"toa-bedrock": 5}


def test_the_master_key_is_refused_a_name_that_several_teams_own(rig: Rig) -> None:
    """`team-split` has a Vertex deployment owned by team A and a Bedrock deployment owned by team B. The master
    key calls it.

    The caller gets HTTP 400 whose message contains `matches deployments from multiple teams` and names
    `team-split`, and the peer receives nothing for either deployment. UNVERIFIED: the exact text the proxy wraps
    around the core's `model name `team-split` matches deployments from multiple teams`.
    """
    response: Final = chat(rig.gateway, "team-split")
    assert response.status_code == 400, response.text
    assert "matches deployments from multiple teams" in error_message(response), response.text
    assert "team-split" in error_message(response), response.text
    assert rig.peer.count("tsp-a-vertex") == 0
    assert rig.peer.count("tsp-b-bedrock") == 0


def test_each_team_reaches_only_its_own_deployment_of_a_shared_name(rig: Rig) -> None:
    """`team-split` again; a key of team A and a key of team B each send ten requests.

    Team A's requests must all reach Vertex and team B's all reach Bedrock.
    """
    with rig.gateway.scenario() as scenario:
        scenario.team(team_id=TEAM_A)
        scenario.team(team_id=TEAM_B)
        key_a: Final = scenario.key(team_id=TEAM_A)
        key_b: Final = scenario.key(team_id=TEAM_B)
        assert served_many(rig.gateway, "team-split", 10, key=key_a) == {"tsp-a-vertex": 10}
        assert served_many(rig.gateway, "team-split", 10, key=key_b) == {"tsp-b-bedrock": 10}


def test_a_team_member_is_shuffled_across_its_own_and_public_deployments(rig: Rig) -> None:
    """`team-shared` has a public OpenAI deployment and a team A Vertex deployment, both `weight: 1`. A team A key
    sends forty requests.

    Both deployments must serve at least five: the team's own deployment is added to the public pool, it does not
    replace it.
    """
    with rig.gateway.scenario() as scenario:
        scenario.team(team_id=TEAM_A)
        key: Final = scenario.key(team_id=TEAM_A)
        counts: Final = served_many(rig.gateway, "team-shared", 40, key=key)
    assert counts["tsh-public-openai"] >= 5, counts
    assert counts["tsh-a-vertex"] >= 5, counts


def test_a_team_key_cannot_reach_another_teams_only_deployment(rig: Rig) -> None:
    """`team-only-a` is served only by team A's Bedrock deployment. A key of team B calls it.

    The request must not succeed and the peer must receive nothing for the deployment. UNVERIFIED: the status,
    which is 429 `No deployments available` if the request reaches the router with an empty pool, or a 4xx if the
    proxy's model access check refuses it first; the test pins only that no call is made.
    """
    with rig.gateway.scenario() as scenario:
        scenario.team(team_id=TEAM_B)
        key: Final = scenario.key(team_id=TEAM_B)
        before: Final = rig.peer.count("toa-bedrock")
        response: Final = chat(rig.gateway, "team-only-a", key=key)
    assert response.status_code != 200, response.text
    assert 400 <= response.status_code < 500, response.text
    assert rig.peer.count("toa-bedrock") == before
