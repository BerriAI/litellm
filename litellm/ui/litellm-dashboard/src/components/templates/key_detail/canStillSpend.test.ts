import { describe, expect, it } from "vitest";

import type { AppliedBudget } from "@/components/shared/appliedBudgets";

import { canStillSpend, usedUpBudget } from "./canStillSpend";

const budget = (overrides: Partial<AppliedBudget>): AppliedBudget => ({
  scope: "Key",
  name: "support-bot",
  maxBudget: 100,
  spend: 0,
  budgetDuration: null,
  ...overrides,
});

describe("canStillSpend", () => {
  it("reports no budget when nothing limits the key", () => {
    expect(canStillSpend([], false)).toEqual({ kind: "noBudget" });
  });

  it("reports unmeasured when no budget reports its spend", () => {
    expect(canStillSpend([budget({ scope: "User", spend: null })], false)).toEqual({ kind: "unmeasured" });
  });

  it("picks the budget with the fewest dollars left, not the highest percent used", () => {
    const key = budget({ scope: "Key", maxBudget: 500, spend: 312.4 });
    const memberLimit: Partial<AppliedBudget> = { scope: "Team member", name: "Support", maxBudget: 600, spend: 412.8 };
    const orgLimit: Partial<AppliedBudget> = {
      scope: "Organization",
      name: "Success",
      maxBudget: 100000,
      spend: 99900,
    };
    const member = budget(memberLimit);
    const org = budget(orgLimit);

    const result = canStillSpend([key, member, org], false);

    expect(result).toMatchObject({ kind: "measured", budget: org, remaining: 100 });
  });

  it("chooses a measured budget over one that does not report spend", () => {
    const user = budget({ scope: "User", maxBudget: 10, spend: null });
    const team = budget({ scope: "Team", maxBudget: 2000, spend: 684.12 });

    const result = canStillSpend([user, team], false);

    expect(result).toMatchObject({ kind: "measured", budget: team });
    expect(result.kind === "measured" && result.remaining).toBeCloseTo(1315.88);
  });

  it("never reports a negative amount for an overspent budget", () => {
    const result = canStillSpend([budget({ maxBudget: 100, spend: 140 })], false);

    expect(result).toMatchObject({ kind: "measured", remaining: 0, spend: 140 });
  });

  it("marks only the key's own budget as throttling when the key throttles on budget", () => {
    const key = budget({ scope: "Key", maxBudget: 100, spend: 90 });
    const team = budget({ scope: "Team", maxBudget: 100, spend: 95 });

    expect(canStillSpend([key], true)).toMatchObject({ throttles: true });
    expect(canStillSpend([key, team], true)).toMatchObject({ budget: team, throttles: false });
    expect(canStillSpend([key], false)).toMatchObject({ throttles: false });
  });
});

describe("usedUpBudget", () => {
  it("returns the first budget that is used up and blocks requests", () => {
    const key = budget({ scope: "Key", maxBudget: 500, spend: 455 });
    const memberLimit: Partial<AppliedBudget> = { scope: "Team member", name: "Support", maxBudget: 600, spend: 600 };
    const member = budget(memberLimit);

    expect(usedUpBudget([key, member], false)).toBe(member);
  });

  it("ignores a used-up key budget that throttles instead of blocking", () => {
    const key = budget({ scope: "Key", maxBudget: 100, spend: 120 });

    expect(usedUpBudget([key], true)).toBeNull();
    expect(usedUpBudget([key], false)).toBe(key);
  });

  it("ignores budgets that do not report spend", () => {
    expect(usedUpBudget([budget({ scope: "User", spend: null })], false)).toBeNull();
  });
});
