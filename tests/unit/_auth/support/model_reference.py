import re
from dataclasses import replace
from typing import Final

from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts


def _pattern(model: str, pattern: str) -> bool:
    return "*" in pattern and re.match("^" + pattern.replace("*", ".*") + "$", model) is not None


def helper(model: str, facts: ModelAccessFacts, *, team_aliases: bool = True) -> bool:
    ordinary: Final = tuple(grant for grant in facts.grants if grant != "all-team-models")
    expanded: Final = tuple(dict.fromkeys((*ordinary, *facts.registry.names)))
    grants: Final = (
        (expanded or facts.grants)
        if "all-team-models" in facts.grants and facts.team_id is not None and facts.registry.present
        else facts.grants
    )
    record: Final = facts.registry.records.get(model)
    groups: Final = record.access_groups if record is not None else ()
    if facts.registry.present and any(grant in groups for grant in grants):
        return True
    filtered: Final = tuple(grant for grant in grants if grant not in groups)
    if team_aliases and model in facts.aliases.team:
        return True
    if any(_pattern(model, grant) for grant in filtered):
        return True
    provider: Final = record.provider if record is not None else None
    if (
        provider is not None
        and (provider.stripped_model != model or "/" not in model)
        and any(_pattern(f"{provider.provider}/{provider.stripped_model}", grant) for grant in filtered)
    ):
        return True
    return not grants or "*" in filtered or "all-proxy-models" in filtered or model in filtered


def _single(model: str, facts: ModelAccessFacts) -> ModelAccessDecision:
    after_team: Final = facts.aliases.team.get(model, model)
    after_key: Final = facts.aliases.key.get(after_team, after_team)
    after_global: Final = facts.aliases.global_.get(after_key, after_key)
    dispatched: Final = facts.aliases.key.get(after_global, after_global)
    key_applied: Final = after_key != after_team or dispatched != after_global
    alias: Final = (
        facts.aliases.global_.get(model) if model in facts.aliases.global_ else facts.registry.aliases.get(model)
    )
    parent: Final = facts.registry.compaction_parents.get(model)
    potential: Final = (
        (dispatched,)
        if key_applied
        else (model, *((parent,) if parent is not None else ()), *((alias,) if alias else ()))
    )
    if any(helper(candidate, facts) for candidate in potential):
        return ModelAccessDecision("Allow")
    record: Final = facts.registry.records.get(model)
    public: Final = (
        record.deployment_public_name if record is not None and record.deployment_team == facts.team_id else None
    )
    owning: Final = (
        (
            *((public,) if public else ()),
            *(alias for alias, target in facts.aliases.team.items() if target == model),
        )
        if not key_applied
        and facts.registry.present
        and facts.team_id is not None
        and model.startswith(f"model_name_{facts.team_id}_")
        and record is not None
        and record.deployment_team == facts.team_id
        else ()
    )
    if any(helper(candidate, facts, team_aliases=False) for candidate in owning):
        return ModelAccessDecision("Allow")
    return ModelAccessDecision("Deny", model, facts.grants, facts.object_type)


def _list_item(model: str, facts: ModelAccessFacts) -> ModelAccessDecision:
    decision: Final = model_access(
        replace(facts, requested=(model,), fallback_depth=facts.fallback_depth + 1, is_list=False)
    )
    return replace(decision, model=model) if decision.kind == "Depth" else decision


def model_access(facts: ModelAccessFacts) -> ModelAccessDecision:
    if facts.fallback_depth >= facts.max_depth:
        return ModelAccessDecision("Depth")
    decisions: Final = (
        _list_item(model, facts) if facts.is_list else _single(model, facts) for model in facts.requested
    )
    return next((decision for decision in decisions if decision.kind != "Allow"), ModelAccessDecision("Allow"))
