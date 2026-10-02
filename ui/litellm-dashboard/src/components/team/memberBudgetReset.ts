import type { components } from "@/lib/http/schema";
import type { TeamMembership } from "./TeamInfo";

export type TeamUpdatePayload = components["schemas"]["UpdateTeamRequest"];

export interface MemberBudgetResetPending {
  readonly teamId: string;
  readonly updateData: TeamUpdatePayload;
  readonly memberCount: number;
  readonly newBudget: number;
}

type MembershipBudgetRow = Pick<TeamMembership, "user_id" | "budget_source"> & {
  litellm_budget_table?: Pick<TeamMembership["litellm_budget_table"], "max_budget"> | null;
};

export const customBudgetMemberUserIds = (memberships: readonly MembershipBudgetRow[]): string[] =>
  memberships
    .filter((m) => m.budget_source === "custom" && m.litellm_budget_table?.max_budget != null)
    .map((m) => m.user_id);

export type MemberBudgetUpdateMode = NonNullable<
  components["schemas"]["UpdateTeamRequest"]["team_member_budget_update_mode"]
>;

export const isMemberBudgetChanging = (
  nextBudget: number | string | null | undefined,
  previousBudget: number | null | undefined,
): boolean => {
  const amount = Number(nextBudget);
  if (!Number.isFinite(amount) || amount <= 0) return false;
  return amount !== previousBudget;
};

export const memberBudgetUpdateMessage = (count: number): string =>
  `Team settings updated. ${count} member ${count === 1 ? "budget now follows" : "budgets now follow"} the team default amount`;
