import type { TeamMembership } from "./TeamInfo";

export const MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES = 500;

export interface MemberBudgetResetPending {
  readonly teamId: string;
  readonly updateData: Record<string, unknown>;
  readonly userIds: readonly string[];
  readonly newBudget: number;
}

type MembershipBudgetRow = Pick<TeamMembership, "user_id" | "budget_source"> & {
  litellm_budget_table?: Pick<TeamMembership["litellm_budget_table"], "max_budget"> | null;
};

export const customBudgetMemberUserIds = (memberships: readonly MembershipBudgetRow[]): string[] =>
  memberships
    .filter((m) => m.budget_source === "custom" && m.litellm_budget_table?.max_budget != null)
    .map((m) => m.user_id);

export const shouldPromptMemberBudgetReset = (
  nextBudget: number | undefined,
  previousBudget: number | null | undefined,
  customBudgetUserIds: string[],
): boolean => {
  const budgetChanged = typeof nextBudget === "number" && nextBudget > 0 && nextBudget !== previousBudget;
  return budgetChanged && customBudgetUserIds.length > 0;
};

export const chunk = <T>(items: readonly T[], size: number): T[][] => {
  if (size <= 0) throw new RangeError("chunk size must be positive");
  return Array.from({ length: Math.ceil(items.length / size) }, (_, i) => items.slice(i * size, i * size + size));
};

export const pluralize = (count: number, singular: string, plural: string): string => (count === 1 ? singular : plural);
