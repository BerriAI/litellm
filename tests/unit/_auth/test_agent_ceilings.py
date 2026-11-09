from collections.abc import Callable
from itertools import chain
from typing import Final

import pytest

from litellm._auth.agent_ceilings import AgentCeilingDecision as Decision
from litellm._auth.agent_ceilings import AgentCeilingFacts as Facts
from litellm._auth.agent_ceilings import AgentCeilingGroup as Group
from litellm._auth.agent_ceilings import AgentGroupLoadDecision as LoadDecision
from litellm._auth.agent_ceilings import AgentGroupLoadFacts as LoadFacts
from litellm._auth.agent_ceilings import agent_ceiling, agent_group_load
from litellm._auth.failure import NativeAuthFailure


def reference(facts: Facts) -> Decision:
    if facts.ids is None:
        return Decision("ReadIds")
    if not facts.ids:
        return Decision("Complete")
    if facts.groups is None:
        return Decision("ReadGroups", authoritative=facts.check_db_only)
    indices: Final = tuple(index for index, group in enumerate(facts.groups) if group.status == "Found")
    return Decision(
        "Complete",
        present=True,
        ids=tuple(name for name, group in zip(facts.ids, facts.groups, strict=True) if group.status != "Missing"),
        models=frozenset(chain.from_iterable(facts.groups[index].models for index in indices)),
        mcp_server_ids=frozenset(chain.from_iterable(facts.groups[index].mcp_server_ids for index in indices)),
        agent_ids=frozenset(chain.from_iterable(facts.groups[index].agent_ids for index in indices)),
        indices=indices,
    )


def load_reference(facts: LoadFacts) -> LoadDecision:
    if not facts.database_present:
        return LoadDecision("DatabaseUnavailable")
    if facts.status is None:
        return LoadDecision("Read")
    if facts.status == "Success":
        return LoadDecision("Return")
    return LoadDecision("Raise" if facts.status == "NotFound" or facts.check_db_only else "HttpUnavailable")


@pytest.mark.parametrize("policy", (reference, agent_ceiling))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (Facts(), Decision("ReadIds")),
        (Facts(()), Decision("Complete")),
        (Facts(("group",)), Decision("ReadGroups")),
        (Facts(("group",), check_db_only=True), Decision("ReadGroups", authoritative=True)),
        (Facts(("group",), (Group("Missing"),)), Decision("Complete", present=True)),
        (Facts(("group",), (Group("Unavailable"),)), Decision("Complete", present=True, ids=("group",))),
        (
            Facts(
                ("one", "two", "one"),
                (Group("Found", ("model",), ("server",), ("agent",)), Group("Missing"), Group("Found", ("model",))),
            ),
            Decision(
                "Complete",
                present=True,
                ids=("one", "one"),
                models=frozenset(("model",)),
                mcp_server_ids=frozenset(("server",)),
                agent_ids=frozenset(("agent",)),
                indices=(0, 2),
            ),
        ),
    ),
)
def test_ceiling_union_and_missing_vs_unavailable(
    policy: Callable[[Facts], Decision], facts: Facts, expected: Decision
) -> None:
    assert policy(facts) == expected


@pytest.mark.parametrize("policy", (load_reference, agent_group_load))
@pytest.mark.parametrize("database", (False, True))
@pytest.mark.parametrize("strict", (False, True))
@pytest.mark.parametrize("status", (None, "Success", "NotFound", "HttpError"))
def test_group_read_error_policy(
    policy: Callable[[LoadFacts], LoadDecision], database: bool, strict: bool, status: str | None
) -> None:
    facts: Final = LoadFacts(database, strict, status)  # pyright: ignore[reportArgumentType]  # parameter table covers the complete typed status domain
    assert policy(facts) == load_reference(facts)


def test_mismatched_group_answers_are_terminal() -> None:
    with pytest.raises(NativeAuthFailure):
        agent_ceiling(Facts(("group",), ()))
