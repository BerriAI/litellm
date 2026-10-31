import { z } from "zod";
import type { BudgetWindowEntry } from "../key_team_helpers/BudgetWindowsEditor";

const budgetWindowEntriesSchema = z.array(
  z.object({
    budget_duration: z.enum(["1h", "24h", "7d", "30d"]),
    max_budget: z.number().nullable(),
  }),
);

export const memberBudgetWindowsSchema = budgetWindowEntriesSchema.superRefine((windows, context) => {
  windows.forEach((window, index) => {
    if (window.max_budget === null) {
      context.addIssue({ code: "custom", message: "Enter a budget" });
    } else if (window.max_budget < 0) {
      context.addIssue({ code: "custom", message: "Enter a non-negative budget" });
    }
    if (windows.slice(0, index).some((other) => other.budget_duration === window.budget_duration)) {
      context.addIssue({
        code: "custom",
        message: "Use each budget duration only once",
      });
    }
  });
});

export const isMemberBudgetWindows = (value: unknown): value is MemberBudgetWindow[] =>
  budgetWindowEntriesSchema.safeParse(value).success;

export type MemberBudgetWindow = z.infer<typeof memberBudgetWindowsSchema>[number];

export const effectiveMemberBudgetWindows = (
  defaults: BudgetWindowEntry[] | null | undefined,
  override: BudgetWindowEntry[] | null | undefined,
): BudgetWindowEntry[] => (override?.length ? override : defaults ?? []);

export const memberBudgetWindowsUpdate = (
  windows: BudgetWindowEntry[] | undefined,
  previous?: BudgetWindowEntry[] | null,
): BudgetWindowEntry[] | undefined => {
  if (windows?.length) return windows;
  if (previous?.length) return [];
  return undefined;
};
