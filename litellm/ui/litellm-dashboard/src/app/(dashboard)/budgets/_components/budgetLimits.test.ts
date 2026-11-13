import { describe, expect, it } from "vitest";
import { budgetFormPayload, COMBINED_BUDGET_DURATION } from "./budgetLimits";

const combined = {
  budget_id: "both",
  budget_duration: COMBINED_BUDGET_DURATION,
  max_budget: 0,
  annual_budget: 100,
  rpm_limit: 20,
};
const limits = [
  { budget_duration: "1mo", max_budget: 0, reset_at: "2027-01-01T00:00:00Z" },
  { budget_duration: "12mo", max_budget: 100 },
];

describe("budgetFormPayload", () => {
  it.each(["30d", "12mo"])("preserves the scalar representation for %s", (duration) => {
    expect(budgetFormPayload({ budget_id: "single", max_budget: 12.345, budget_duration: duration })).toEqual({
      budget_id: "single",
      max_budget: 12.35,
      budget_duration: duration,
    });
  });

  it("creates independent caps without rounding a sub-cent limit", () => {
    expect(budgetFormPayload({ ...combined, annual_budget: 0.0001 })).toEqual({
      budget_id: "both",
      rpm_limit: 20,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 0 },
        { budget_duration: "12mo", max_budget: 0.0001 },
      ],
    });
  });

  it("does not resend caps for a rate-only edit", () => {
    expect(budgetFormPayload(combined, { rpm_limit: true }, limits)).toEqual({ budget_id: "both", rpm_limit: 20 });
  });

  it("preserves the latest untouched cap and reset anchor when editing a combined budget", () => {
    expect(
      budgetFormPayload({ ...combined, max_budget: 90, annual_budget: 200 }, { annual_budget: true }, limits),
    ).toEqual({
      budget_id: "both",
      budget_limits: [limits[0], { ...limits[1], max_budget: 200 }],
    });
  });

  it("clears the scalar representation when switching to both", () => {
    const expected = {
      budget_id: "both",
      max_budget: null,
      budget_duration: null,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 0 },
        { budget_duration: "12mo", max_budget: 100 },
      ],
    };
    expect(budgetFormPayload(combined, { budget_duration: true, annual_budget: true })).toEqual(expected);
  });

  it.each(["30d", "12mo"])("writes an unchanged cap when switching from both to %s", (duration) => {
    const expected = {
      budget_id: "both",
      max_budget: 0,
      budget_duration: duration,
      budget_limits: null,
    };
    expect(budgetFormPayload({ ...combined, budget_duration: duration }, { budget_duration: true }, limits)).toEqual(
      expected,
    );
  });

  it("preserves explicit null clears", () => {
    expect(
      budgetFormPayload(
        { budget_id: "single", max_budget: null, budget_duration: null },
        { max_budget: true, budget_duration: true },
      ),
    ).toEqual({
      budget_id: "single",
      max_budget: null,
      budget_duration: null,
    });
  });

  it("preserves other existing windows when changing a calendar cap", () => {
    const hourly = { budget_duration: "1h", max_budget: 5 };
    expect(budgetFormPayload(combined, { annual_budget: true }, [...limits, hourly])).toEqual({
      budget_id: "both",
      budget_limits: [hourly, ...limits],
    });
  });
});
