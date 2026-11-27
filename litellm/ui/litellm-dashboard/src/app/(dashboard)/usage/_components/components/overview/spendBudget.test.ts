import { describe, expect, it } from "vitest";
import i18n from "@/i18n/i18n";
import type { UserInfoV2Response } from "@/components/networking";
import { spendBudgetFromUser, spendBudgetHint } from "./spendBudget";

const t = i18n.t.bind(i18n);
const user = (overrides: Partial<UserInfoV2Response>) =>
  ({
    spend: 12.4,
    max_budget: 100,
    budget_duration: "24h",
    budget_reset_at: "2026-11-27T00:00:00Z",
    ...overrides,
  }) as UserInfoV2Response;

describe("spendBudgetFromUser", () => {
  it("returns no budget when the user has no limit or has not loaded", () => {
    expect(spendBudgetFromUser(undefined)).toBeNull();
    expect(spendBudgetFromUser(user({ max_budget: null }))).toBeNull();
  });

  it("uses the rollover-adjusted limit when there is one", () => {
    expect(spendBudgetFromUser(user({ effective_max_budget: 130 }))?.maxBudget).toBe(130);
  });

  it("treats an empty duration as no duration", () => {
    expect(spendBudgetFromUser(user({ budget_duration: "" }))?.budgetDuration).toBeNull();
  });
});

describe("spendBudgetHint", () => {
  it("pairs a periodic limit with the spend for the current period, not the range total", () => {
    expect(spendBudgetHint(spendBudgetFromUser(user({})), t)).toBe("$12.40 of $100.00 this period (daily)");
  });

  it("keeps the lifetime wording when the budget never resets", () => {
    expect(spendBudgetHint(spendBudgetFromUser(user({ budget_duration: null })), t)).toBe("of $100.00 budget");
  });

  it("names a non-standard duration as configured", () => {
    expect(spendBudgetHint(spendBudgetFromUser(user({ budget_duration: "2d" })), t)).toBe(
      "$12.40 of $100.00 this period (2d)",
    );
  });

  it("shows a zero limit as a limit rather than as no budget", () => {
    expect(spendBudgetHint(spendBudgetFromUser(user({ max_budget: 0, spend: 0 })), t)).toBe(
      "$0.00 of $0.00 this period (daily)",
    );
  });

  it("says no budget is set when there is none", () => {
    expect(spendBudgetHint(null, t)).toBe("No budget set");
  });
});
