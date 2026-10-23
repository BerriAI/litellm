import { describe, expect, it } from "vitest";
import {
  budgetResetPeriodKey,
  previewMemberBudgetBackfill,
  shouldPromptMemberBudgetReset,
  type MemberBudgetBackfillPreview,
} from "./memberBudgetReset";

const custom = (max_budget: number | null, budget_duration: string | null = "30d") => ({
  budget_source: "custom" as const,
  litellm_budget_table: { max_budget, budget_duration },
});

describe("budgetResetPeriodKey", () => {
  it.each([
    ["7d", "1w", true],
    ["24h", "1d", true],
    ["daily", "1d", true],
    ["60s", "1m", true],
    ["monthly", "30d", true],
    [" 30d ", "30d", true],
    ["1mo", "1mo", true],
    [null, undefined, true],
    ["1mo", "30d", false],
    ["2mo", "1mo", false],
    ["7d", "1d", false],
    ["30d", null, false],
    ["garbage", "garbage", true],
    ["garbage", "30d", false],
  ])("treats %s and %s as the same schedule: %s", (first, second, same) => {
    expect(budgetResetPeriodKey(first) === budgetResetPeriodKey(second)).toBe(same);
  });
});

describe("previewMemberBudgetBackfill", () => {
  it("counts custom budgets on each side of the new default that reset on its schedule", () => {
    const preview = previewMemberBudgetBackfill(
      [custom(5), custom(10, "1mo"), custom(80, "30d"), custom(90, "monthly"), custom(3, "7d"), custom(70, null)],
      20,
      "30d",
    );

    expect(preview).toEqual({ below: 1, above: 2, otherResetPeriod: 3 });
  });

  it("ignores members on the default, members without a cap of their own and caps already at the new default", () => {
    const preview = previewMemberBudgetBackfill(
      [
        { budget_source: "team_default" as const, litellm_budget_table: { max_budget: 5, budget_duration: "30d" } },
        { budget_source: "none" as const },
        custom(null),
        { budget_source: "custom" as const, litellm_budget_table: null },
        custom(20),
        custom(20, "7d"),
      ],
      20,
      "30d",
    );

    expect(preview).toEqual({ below: 0, above: 0, otherResetPeriod: 0 });
  });

  it("matches a default that never resets only to caps that never reset", () => {
    expect(previewMemberBudgetBackfill([custom(5, null), custom(5, "30d")], 20, null)).toEqual({
      below: 1,
      above: 0,
      otherResetPeriod: 1,
    });
  });
});

describe("shouldPromptMemberBudgetReset", () => {
  const movable: MemberBudgetBackfillPreview = { below: 1, above: 0, otherResetPeriod: 0 };
  const nothingMovable: MemberBudgetBackfillPreview = { below: 0, above: 0, otherResetPeriod: 4 };

  it("prompts when the default changes and some member could move onto it", () => {
    expect(shouldPromptMemberBudgetReset(10, 5, movable)).toBe(true);
    expect(shouldPromptMemberBudgetReset(10, null, { below: 0, above: 1, otherResetPeriod: 0 })).toBe(true);
  });

  it("does not prompt when the default is unchanged, cleared or zero", () => {
    expect(shouldPromptMemberBudgetReset(10, 10, movable)).toBe(false);
    expect(shouldPromptMemberBudgetReset(undefined, 10, movable)).toBe(false);
    expect(shouldPromptMemberBudgetReset(0, 10, movable)).toBe(false);
  });

  it("does not prompt when only members on another schedule differ", () => {
    expect(shouldPromptMemberBudgetReset(10, 5, nothingMovable)).toBe(false);
  });
});
