import { describe, expect, it } from "vitest";
import { effectiveMemberBudgetWindows, memberBudgetWindowsSchema } from "./memberBudgetWindowSchema";

const windows = [
  { budget_duration: "24h", max_budget: 20 },
  { budget_duration: "30d", max_budget: 200 },
];

describe("member budget windows", () => {
  it("should accept simultaneous daily and monthly limits and preserve zero-dollar caps", () => {
    expect(memberBudgetWindowsSchema.parse(windows)).toEqual(windows);
    expect(memberBudgetWindowsSchema.parse([{ ...windows[0], max_budget: 0 }])).toEqual([
      { ...windows[0], max_budget: 0 },
    ]);
  });

  it.each([null, -1, Infinity, NaN])("should reject an invalid budget of %s", (max_budget) => {
    expect(memberBudgetWindowsSchema.safeParse([{ ...windows[0], max_budget }]).success).toBe(false);
  });

  it("should reject duplicate and unsupported durations", () => {
    expect(memberBudgetWindowsSchema.safeParse([windows[0], windows[0]]).success).toBe(false);
    expect(memberBudgetWindowsSchema.safeParse([{ budget_duration: "tomorrow", max_budget: 20 }]).success).toBe(false);
  });

  it("should inherit defaults when no override is configured or an override is cleared", () => {
    expect(effectiveMemberBudgetWindows(windows, undefined)).toEqual(windows);
    expect(effectiveMemberBudgetWindows(windows, [])).toEqual(windows);
    expect(effectiveMemberBudgetWindows(undefined, undefined)).toEqual([]);
  });

  it("should replace all default windows with the member override", () => {
    const override = [{ budget_duration: "24h", max_budget: 5 }];
    expect(effectiveMemberBudgetWindows(windows, override)).toEqual(override);
  });
});
