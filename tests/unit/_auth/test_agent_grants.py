from dataclasses import replace
from itertools import product
from typing import Final, assert_never

import pytest

from litellm._auth import agent_grants as policy


def _decision(
    action: policy.Action = "Complete", grants: tuple[str, ...] | None = None, team_id: str | None = None
) -> policy.Decision:
    return policy.Decision(action, grants, team_id)


def reference(facts: policy.Facts) -> policy.Decision:
    match facts:
        case policy.Resolve():
            if facts.managed:
                return _decision("ReadManaged")
            if not facts.key_team_loaded:
                return _decision("ReadKeyTeam")
            if not facts.strict and facts.role != "proxy_admin" and facts.default_deny is None:
                return _decision("ReadDefault")
            if (facts.strict or (facts.role != "proxy_admin" and facts.default_deny)) and facts.key_team is None:
                return _decision(grants=())
            if not facts.caller_loaded:
                return _decision("ReadCaller")
            if not facts.ceiling_loaded:
                return _decision("ReadCeiling")
            return _decision(
                grants=facts.ceiling
                if facts.own is None
                else facts.own
                if facts.ceiling is None
                else tuple(item for item in facts.own if item in facts.ceiling)
            )
        case policy.Failure():
            return _decision("Unavailable") if facts.strict else _decision(grants=() if facts.closed else None)
        case policy.OwnerTeam():
            if facts.permission_id_present and not facts.permission_present:
                return _decision("Unavailable") if facts.strict else _decision(grants=())
            return _decision("ReadDeclared")
        case policy.OwnerUnion():
            if any(source.unavailable for source in facts.teams):
                return _decision("Unavailable")
            restrictions: Final = (
                *(source.grants for source in facts.teams),
                *(() for source in facts.withheld if source.unavailable or source.grants is not None),
                *(((),) if facts.unreadable else ()),
            )
            return _decision(
                grants=None
                if restrictions and all(grants is None for grants in restrictions)
                else tuple(sorted({item for grants in restrictions for item in (grants or ())}))
            )
        case policy.BillingTeam():
            restricted: Final = tuple(
                source for source in facts.sources if source.grants is not None and facts.agent_id in source.grants
            )
            selected: Final = restricted or (
                facts.sources
                if all(source.grants is None and not source.unavailable for source in facts.sources)
                else ()
            )
            return _decision(
                team_id=min((source.team_id for source in selected if source.team_id is not None), default=None)
            )
        case policy.Human():
            return _decision(
                grants=tuple(
                    sorted(
                        {
                            item
                            for source in facts.sources
                            if source.team_id is None or source.team_id == facts.team_id
                            for item in (source.grants or ())
                        }
                    )
                )
            )
        case policy.Delegated():
            if any(source.team_id is None and facts.agent_id in (source.grants or ()) for source in facts.sources):
                return _decision(team_id=facts.team_id)
            teams: Final = frozenset(
                source.team_id
                for source in facts.sources
                if source.team_id is not None
                and facts.agent_id in (source.grants or ())
                and (facts.allowed is None or source.team_id in facts.allowed)
            )
            if facts.team_id in teams:
                return _decision(team_id=facts.team_id)
            if not facts.explicit and teams:
                return _decision(team_id=min(teams))
            return _decision("SelectTeam")
        case policy.TeamLookup():
            if not facts.subject:
                return _decision()
            if facts.team_id is None:
                return _decision(
                    "ReadPersonalEligibility"
                    if facts.personal is None
                    else "ReadOwnerTeams"
                    if facts.personal
                    else "Complete"
                )
            if not facts.database or (facts.loaded and not facts.row_present):
                return _decision(grants=() if facts.strict else None)
            return _decision("ReadDeclared" if facts.loaded else "ReadTeam")
        case policy.GroupMatches():
            return _decision(
                grants=tuple(
                    agent.agent_id for agent in facts.agents if any(group in agent.groups for group in facts.groups)
                )
            )
        case _:
            assert_never(facts)


_SOURCES: Final = (
    (),
    (policy.Source("b", None),),
    (policy.Source("b", ("agent",)), policy.Source("a", ("agent", "other"))),
    (policy.Source("b", None, True),),
    (policy.Source(None, ("agent",)),),
    (policy.Source("b", ()),),
)
_RESOLVE: Final = tuple(
    policy.Resolve(
        managed,
        strict,
        role,
        loaded,
        key,
        default,
        caller_loaded=caller,
        own=("agent", "other"),
        ceiling_loaded=ceiling_loaded,
        ceiling=ceiling,
    )
    for managed, strict, role, loaded, key, default, caller, ceiling_loaded, ceiling in product(
        (False, True),
        (False, True),
        (None, "proxy_admin", "internal_user"),
        (False, True),
        (None, (), ("agent",)),
        (None, False, True),
        (False, True),
        (False, True),
        (None, (), ("agent",)),
    )
)
_CASES: Final[tuple[policy.Facts, ...]] = (
    *_RESOLVE,
    *(
        policy.OwnerUnion(teams, withheld, unreadable)
        for teams, withheld, unreadable in product(_SOURCES, _SOURCES, (False, True))
    ),
    *(policy.Failure(strict, closed) for strict, closed in product((False, True), repeat=2)),
    *(policy.OwnerTeam(strict, declared, loaded) for strict, declared, loaded in product((False, True), repeat=3)),
    *(policy.BillingTeam(sources, "agent") for sources in _SOURCES),
    *(policy.Human(sources, team) for sources, team in product(_SOURCES, (None, "a", "b"))),
    *(
        policy.Delegated(sources, "agent", team, explicit, allowed)
        for sources, team, explicit, allowed in product(_SOURCES, (None, "a", "b"), (False, True), (None, (), ("b",)))
    ),
    *(
        policy.TeamLookup(subject, team, database, strict, personal, loaded, row)
        for subject, team, database, strict, personal, loaded, row in product(
            (False, True),
            (None, "", "team"),
            (False, True),
            (False, True),
            (None, False, True),
            (False, True),
            (False, True),
        )
    ),
    policy.GroupMatches((policy.ConfiguredAgent("agent", ("group",)), policy.ConfiguredAgent("other", ())), ("group",)),
)


@pytest.mark.parametrize("facts", _CASES)
def test_agent_grants_match_reference(facts: policy.Facts) -> None:
    assert policy.decide(facts) == reference(facts)


def test_default_deny_precedes_caller_and_ceiling_reads() -> None:
    facts: Final = policy.Resolve(False, False, "internal_user", key_team_loaded=True)
    assert policy.decide(facts).action == "ReadDefault"
    assert policy.decide(replace(facts, default_deny=True)) == _decision(grants=())
    assert policy.decide(replace(facts, default_deny=False)).action == "ReadCaller"
