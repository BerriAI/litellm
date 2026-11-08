import math
from dataclasses import replace
from typing import Final

from litellm._auth.common_budgets import BudgetDecision, BudgetFacts
from litellm._auth.model_budget_data import ModelBudgetDecision, ModelBudgetSnapshot


def common(f: BudgetFacts) -> BudgetDecision:
    limit: Final = f.limit
    if limit is None:
        return BudgetDecision()
    if f.scope == "User" and f.team_key and not f.apply_user_budget_to_team_keys:
        return BudgetDecision()
    if f.scope == "EndUser" and f.info_route:
        return BudgetDecision()
    if f.scope == "Project" and not math.isfinite(limit):
        return BudgetDecision()
    if f.scope == "AccessGroup" and limit <= 0:
        return BudgetDecision()
    if f.scope == "Proxy" and (not limit > 0 or f.skip or not f.llm_route or f.model_list_route):
        return BudgetDecision()
    if f.spend is None:
        return BudgetDecision(read_spend=f.scope not in ("KeySoft", "TeamSoft", "ProjectSoft", "Proxy"))
    if f.scope.endswith("Soft"):
        enabled: Final = {
            "KeySoft": bool(limit),
            "TeamSoft": f.token_present and bool(f.alert_recipients),
            "ProjectSoft": f.token_present,
        }[f.scope]
        return BudgetDecision(alert="soft_budget" if enabled and f.spend >= limit else None)
    denied: Final = (
        not f.spend <= limit
        if f.scope == "Tag"
        else not f.spend < limit
        if f.scope == "AccessGroup"
        else f.spend > limit
        if f.scope == "EndUser"
        else math.isfinite(limit) and (f.spend > limit if f.scope in ("Team", "Proxy") else f.spend >= limit)
    )
    result: Final = BudgetDecision(
        denied=denied,
        alert=(
            "token_budget"
            if f.scope == "Key"
            else "user_budget"
            if f.scope == "User"
            else "team_budget"
            if f.scope == "Team" and denied and f.token_present
            else "project_budget"
            if f.scope == "Project" and denied and f.token_present
            else "organization_budget"
            if f.scope == "Organization" and denied
            else "team_member_threshold"
            if f.scope == "TeamMember" and math.isfinite(limit)
            else None
        ),
    )
    if (
        f.scope == "Key"
        and denied
        and f.throttle_opt_in
        and (f.tpm_limit is not None or f.rpm_limit is not None)
        and f.throttle_percentage is not None
        and 0 < f.throttle_percentage <= 1
    ):
        return replace(result, denied=False, throttle_percentage=f.throttle_percentage)
    return result


def model(f: ModelBudgetSnapshot) -> ModelBudgetDecision:
    local: Final = (f.local_spend or 0.0) + (f.legacy_spend or 0.0)
    current: Final = (
        max(local, f.shared_spend)
        if f.shared_spend is not None
        and f.shared_started_at is not None
        and f.shared_window_seconds is not None
        and f.now < f.shared_started_at + f.shared_window_seconds
        else local
    )
    return ModelBudgetDecision(
        None, f.max_budget is not None and not f.max_budget < 0 and current >= f.max_budget, current
    )
