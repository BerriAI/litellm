import type { TeamData, TeamMembership } from "./TeamInfo";

export const displayedMemberBudget = (
  membership: TeamMembership | undefined,
  teamDefault: TeamData["team_info"]["team_member_budget_table"],
) => {
  const budget = membership?.litellm_budget_table;
  const hasTemporaryGrant = budget?.temp_budget_increase != null && budget.temp_budget_expiry != null;
  const temporaryOnly =
    hasTemporaryGrant &&
    !budget.allowed_models?.length &&
    [
      budget.max_budget,
      budget.soft_budget,
      budget.max_parallel_requests,
      budget.rpm_limit,
      budget.tpm_limit,
      budget.tpd_limit,
      budget.model_max_budget,
      budget.budget_duration,
    ].every((value) => value == null);
  if (temporaryOnly || (!budget && membership?.budget_source === "team_default")) {
    return teamDefault;
  }
  return budget;
};
