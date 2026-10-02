import type { components } from "@/lib/http/schema";
export type MemberBudgetUpdateMode = NonNullable<
  components["schemas"]["UpdateTeamRequest"]["team_member_budget_update_mode"]
>;

export const isMemberBudgetChanging = (
  nextBudget: number | string | null | undefined,
  previousBudget: number | null | undefined,
): boolean => {
  const amount = Number(nextBudget);
  return Number.isFinite(amount) && amount > 0 && amount !== previousBudget;
};

export const memberBudgetUpdateMessage = (count: number): string =>
  `Team settings updated. ${count} member ${count === 1 ? "budget now follows" : "budgets now follow"} the team default amount`;
