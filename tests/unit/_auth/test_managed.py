from dataclasses import replace
from typing import Final

import pytest

from litellm._auth import managed
from tests.unit._auth import reference_policy

AGENT: Final = managed.Agent("agent", True, True, managed.BoundIdentity(True, "revision"), "both")
CONTEXT: Final = managed.Context("agent", "revision", "delegated", "user")


@pytest.mark.parametrize(
    "agent,context",
    (
        (AGENT, CONTEXT),
        (replace(AGENT, enabled=False), None),
        (replace(AGENT, identity=None), CONTEXT),
        (replace(AGENT, identity=managed.BoundIdentity(False, "old")), CONTEXT),
        (AGENT, None),
        (AGENT, replace(CONTEXT, agent_id="other")),
        (AGENT, replace(CONTEXT, binding_revision=None)),
        (AGENT, replace(CONTEXT, binding_revision="old")),
        (replace(AGENT, execution_mode="autonomous"), CONTEXT),
        (AGENT, replace(CONTEXT, user_id=None)),
        (AGENT, replace(CONTEXT, user_id="")),
    ),
)
def test_actor_denial_order_preserves_verified_context_contract(
    agent: managed.Agent, context: managed.Context | None
) -> None:
    facts: Final = managed.Actor(agent, context)
    assert managed.decide(facts) == reference_policy.actor(facts)


def test_actor_read_plan_assigns_policy_before_delegation_grant_read() -> None:
    facts: Final = managed.ActorLookup("agent", True, context=CONTEXT)
    assert managed.decide(facts).action == "ReadAgent"
    loaded: Final = replace(facts, agent_loaded=True, agent=AGENT)
    assert managed.decide(loaded).action == "AssignPolicy"
    assigned: Final = replace(loaded, policy_assigned=True)
    assert managed.decide(assigned).action == "ReadHumanGrants"
    assert managed.decide(replace(assigned, human_grants=())).denial == "HumanNotPermitted"
    assert managed.decide(replace(assigned, human_grants=("agent",))).action == "Allow"
    assert managed.decide(replace(assigned, delegation_verified=True)).action == "Allow"


def test_missing_actor_still_reads_retirement_before_denial() -> None:
    facts: Final = managed.ActorLookup("agent", True, agent_loaded=True, context=CONTEXT)
    assert managed.decide(facts).action == "ReadRetired"
    assert managed.decide(replace(facts, retired=False)).denial == "Removed"
    assert managed.decide(replace(facts, retired=False, context=None)).action == "Allow"
