import type { components } from "@/lib/http/schema";
import type { TeamMembership } from "./TeamInfo";

export type TeamUpdatePayload = components["schemas"]["UpdateTeamRequest"];

export type MemberBudgetBackfill = NonNullable<TeamUpdatePayload["team_member_budget_backfill"]>;

export interface MemberBudgetBackfillResult {
  readonly moved_to_default: number;
  readonly skipped_other_reset_period: number;
}

export interface MemberBudgetBackfillPreview {
  readonly below: number;
  readonly above: number;
  readonly otherResetPeriod: number;
}

export interface MemberBudgetResetPending {
  readonly updateData: TeamUpdatePayload;
  readonly newBudget: number;
  readonly preview: MemberBudgetBackfillPreview;
}

type MembershipBudgetRow = Pick<TeamMembership, "budget_source"> & {
  litellm_budget_table?: Pick<TeamMembership["litellm_budget_table"], "max_budget" | "budget_duration"> | null;
};

const FIXED_UNIT_SECONDS: Readonly<Record<string, number>> = { s: 1, m: 60, h: 3600, d: 86400, w: 604800 };
const DURATION_WORD_ALIASES: Readonly<Record<string, string>> = {
  hourly: "1h",
  daily: "24h",
  weekly: "7d",
  monthly: "30d",
};

export const budgetResetPeriodKey = (duration: string | null | undefined): string | null => {
  if (duration == null) return null;
  const normalized = (DURATION_WORD_ALIASES[duration.trim().toLowerCase()] ?? duration).trim();
  const parsed = /^(\d+)(mo|[smhdw])$/.exec(normalized);
  if (parsed === null) return normalized;
  const value = Number(parsed[1]);
  const unit = parsed[2];
  return unit === "mo" ? `${value}mo` : `${value * FIXED_UNIT_SECONDS[unit]}s`;
};

export const previewMemberBudgetBackfill = (
  memberships: readonly MembershipBudgetRow[],
  newBudget: number,
  newBudgetDuration: string | null | undefined,
): MemberBudgetBackfillPreview => {
  const schedule = budgetResetPeriodKey(newBudgetDuration);
  const differing = memberships.flatMap((m) => {
    const budget = m.litellm_budget_table;
    if (m.budget_source !== "custom" || budget?.max_budget == null || budget.max_budget === newBudget) return [];
    return [{ amount: budget.max_budget, sameSchedule: budgetResetPeriodKey(budget.budget_duration) === schedule }];
  });
  const comparable = differing.filter((m) => m.sameSchedule);
  return {
    below: comparable.filter((m) => m.amount < newBudget).length,
    above: comparable.filter((m) => m.amount > newBudget).length,
    otherResetPeriod: differing.length - comparable.length,
  };
};

export const shouldPromptMemberBudgetReset = (
  nextBudget: number | undefined,
  previousBudget: number | null | undefined,
  preview: MemberBudgetBackfillPreview,
): boolean => {
  const budgetChanged = typeof nextBudget === "number" && nextBudget > 0 && nextBudget !== previousBudget;
  return budgetChanged && preview.below + preview.above > 0;
};

export const pluralize = (count: number, singular: string, plural: string): string => (count === 1 ? singular : plural);
