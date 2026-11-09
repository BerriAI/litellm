from dataclasses import replace
from typing import Final

from litellm._auth.fallback_access import FallbackAuthorizationFacts, GroupedModelScope
from litellm._auth.key_model_access import KeyAndFallbackModelAccessFacts
from litellm._auth.model_access import ModelAccessDecision
from tests.unit._auth.support.model_reference import model_access


def grouped(scope: GroupedModelScope) -> ModelAccessDecision:
    direct: Final = model_access(scope.facts)
    if direct.kind == "Allow" or not scope.group_ids:
        return direct
    if scope.group_models is None:
        raise AssertionError("unread resource groups")
    return model_access(replace(scope.facts, grants=scope.group_models)) if scope.group_models else direct


def initial_access(facts: KeyAndFallbackModelAccessFacts) -> ModelAccessDecision | None:
    if facts.config_nonempty or "all-team-models" in facts.key_models:
        return None
    if not facts.primary_loaded:
        raise AssertionError("unread primary")
    if facts.primary is not None:
        primary: Final = grouped(facts.primary)
        if primary.kind != "Allow":
            return primary
    if facts.fallback_names is None:
        raise AssertionError("unread fallbacks")
    for index, _ in enumerate(facts.fallback_names):
        if (result := grouped(facts.fallback_scopes[index])).kind != "Allow":
            return result
        if facts.routing[index] != "Routed":
            raise AssertionError("routing failure should be projected by the effect")
    return None


def fallback_access(facts: FallbackAuthorizationFacts) -> bool:
    if facts.refuse_agent_keys and facts.agent_id is not None:
        return False
    if facts.key is None:
        raise AssertionError("unread key scope")
    if grouped(facts.key).kind != "Allow":
        return False
    if facts.team is not None and grouped(facts.team).kind != "Allow":
        assigned: Final = replace(facts.team.facts, grants=facts.assigned_models or ())
        if not facts.assigned_models or model_access(assigned).kind != "Allow":
            return False
    return all(
        scope is None or model_access(scope).kind == "Allow"
        for scope in (facts.embedded_team, facts.member, facts.project)
    )
