import type { ParseKeys } from "i18next";

import type { AppliedBudget } from "@/components/shared/appliedBudgets";

export interface TextRef {
  readonly key: ParseKeys;
  readonly values?: Readonly<Record<string, string | number>>;
}

export const SCOPE_KEYS: Readonly<Record<AppliedBudget["scope"], string>> = {
  Key: "key",
  Team: "team",
  "Team member": "teamMember",
  Organization: "organization",
  User: "user",
};

export const ownerText = (budget: AppliedBudget, capitalized: boolean): TextRef => ({
  key: `keyDetail.overview.canStillSpend.${capitalized ? "ownerTitle" : "owner"}.${SCOPE_KEYS[budget.scope]}` as ParseKeys,
  values: { name: budget.name },
});

export const headroomKey = (usedUp: boolean, throttles: boolean): ParseKeys => {
  if (usedUp) return "keyDetail.overview.canStillSpend.usedUp";
  return throttles
    ? "keyDetail.overview.canStillSpend.beforeThrottles"
    : "keyDetail.overview.canStillSpend.beforeBlocks";
};

export const resetSentence = (budget: AppliedBudget, keyResetTime: string | null): TextRef => {
  const resetTime = budget.scope === "Key" ? keyResetTime : null;
  if (resetTime) return { key: "keyDetail.overview.canStillSpend.resetsAt", values: { time: resetTime } };
  if (budget.budgetDuration) {
    return { key: "keyDetail.overview.canStillSpend.resetsEvery", values: { duration: budget.budgetDuration } };
  }
  return { key: "keyDetail.overview.canStillSpend.noReset" };
};

export const resetCaption = (budget: AppliedBudget, keyResetTime: string | null): TextRef => {
  const resetTime = budget.scope === "Key" ? keyResetTime : null;
  if (resetTime) return { key: "keyDetail.overview.budgets.resetsAt", values: { time: resetTime } };
  if (budget.budgetDuration) {
    return { key: "shared.appliedBudgets.resetsEvery", values: { duration: budget.budgetDuration } };
  }
  return { key: "shared.appliedBudgets.noReset" };
};

export const meterTone = (percent: number): "over" | "warning" | "default" => {
  if (percent >= 100) return "over";
  return percent >= 80 ? "warning" : "default";
};
