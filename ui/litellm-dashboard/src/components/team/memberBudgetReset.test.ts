import { describe, expect, it } from "vitest";
import {
  chunk,
  customBudgetMemberUserIds,
  MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES,
  pluralize,
  shouldPromptMemberBudgetReset,
} from "./memberBudgetReset";

describe("customBudgetMemberUserIds", () => {
  const customRow = (user_id: string, max_budget: number | null = 50) => ({
    user_id,
    budget_source: "custom" as const,
    litellm_budget_table: { max_budget },
  });

  it("returns only members whose budget_source is custom with a private cap", () => {
    const memberships = [
      customRow("u-custom"),
      { user_id: "u-default", budget_source: "team_default" as const },
      { user_id: "u-none", budget_source: "none" as const },
    ];

    expect(customBudgetMemberUserIds(memberships)).toEqual(["u-custom"]);
  });

  it("skips a custom row whose cap is null since it already inherits the default", () => {
    const memberships = [customRow("u-rate-limits-only", null), customRow("u-custom")];

    expect(customBudgetMemberUserIds(memberships)).toEqual(["u-custom"]);
  });

  it("skips a custom row that has no budget table at all", () => {
    const memberships = [
      { user_id: "u-no-table", budget_source: "custom" as const, litellm_budget_table: null },
      customRow("u-custom"),
    ];

    expect(customBudgetMemberUserIds(memberships)).toEqual(["u-custom"]);
  });
});

describe("shouldPromptMemberBudgetReset", () => {
  it("prompts when the default changes and custom-budget members exist", () => {
    expect(shouldPromptMemberBudgetReset(10, 5, ["u-1"])).toBe(true);
  });

  it("prompts when a default is set for the first time", () => {
    expect(shouldPromptMemberBudgetReset(10, null, ["u-1"])).toBe(true);
    expect(shouldPromptMemberBudgetReset(10, undefined, ["u-1"])).toBe(true);
  });

  it("does not prompt when the submitted budget is unchanged", () => {
    expect(shouldPromptMemberBudgetReset(10, 10, ["u-1"])).toBe(false);
  });

  it("does not prompt when the budget is cleared or zero", () => {
    expect(shouldPromptMemberBudgetReset(undefined, 10, ["u-1"])).toBe(false);
    expect(shouldPromptMemberBudgetReset(0, 10, ["u-1"])).toBe(false);
  });

  it("does not prompt when no member has a custom budget", () => {
    expect(shouldPromptMemberBudgetReset(10, 5, [])).toBe(false);
  });
});

describe("chunk", () => {
  it("splits selections larger than the bulk endpoint limit", () => {
    const ids = Array.from({ length: MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1 }, (_, i) => `u-${i}`);

    const chunks = chunk(ids, MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES);

    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toHaveLength(MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES);
    expect(chunks[1]).toEqual([`u-${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES}`]);
  });

  it("keeps a selection under the limit in a single chunk", () => {
    expect(chunk(["u-1", "u-2"], MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES)).toEqual([["u-1", "u-2"]]);
  });

  it("returns no chunks for an empty selection", () => {
    expect(chunk([], MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES)).toEqual([]);
  });

  it("rejects a non-positive size instead of dividing by zero", () => {
    expect(() => chunk(["u-1"], 0)).toThrow(RangeError);
  });
});

describe("pluralize", () => {
  it("uses the singular form for exactly one", () => {
    expect(pluralize(1, "budget", "budgets")).toBe("budget");
  });

  it("uses the plural form for zero and for more than one", () => {
    expect(pluralize(0, "budget", "budgets")).toBe("budgets");
    expect(pluralize(3, "budget", "budgets")).toBe("budgets");
  });
});
