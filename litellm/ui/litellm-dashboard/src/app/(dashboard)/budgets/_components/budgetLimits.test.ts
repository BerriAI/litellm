import { describe, expect, it } from "vitest";
import type { components } from "@/lib/http/schema";
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
] satisfies NonNullable<components["schemas"]["BudgetWriteRequest"]["budget_limits"]>;

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

  it.each(["1mo", "12mo"] as const)("updates a single %s window in place, including zero", (duration) => {
    const existing = { budget_duration: duration, max_budget: 100, reset_at: "2027-01-01T00:00:00Z" };
    const values = { budget_id: "single", max_budget: 0, budget_duration: duration === "1mo" ? "30d" : duration };
    expect(budgetFormPayload(values, { max_budget: true }, [existing])).toEqual({
      budget_id: "single",
      budget_limits: [{ ...existing, max_budget: 0 }],
    });
  });

  it("does not resend a single window for a rate-only edit", () => {
    const values = { budget_id: "single", max_budget: 100, budget_duration: "12mo", rpm_limit: 20 };
    expect(budgetFormPayload(values, { rpm_limit: true }, [limits[1]])).toEqual({ budget_id: "single", rpm_limit: 20 });
  });

  it("preserves a second window received on refresh while editing a single monthly cap", () => {
    const values = { budget_id: "single", max_budget: 50, budget_duration: "30d" };
    expect(budgetFormPayload(values, { max_budget: true }, limits)).toEqual({
      budget_id: "single",
      budget_limits: [{ ...limits[0], max_budget: 50 }, limits[1]],
    });
  });

  it("clears the existing single window when the amount is blank", () => {
    const values = { budget_id: "single", max_budget: null, budget_duration: "30d" };
    expect(budgetFormPayload(values, { max_budget: true }, [limits[0]])).toEqual({
      budget_id: "single",
      budget_limits: null,
    });
  });

  it.each([0, null])("edits only the additional scalar cap to %s while preserving calendar caps", (amount) => {
    const values = { ...combined, additional_max_budget: amount, additional_budget_duration: "24h" };
    expect(budgetFormPayload(values, { additional_max_budget: true }, limits)).toEqual({
      budget_id: "both",
      max_budget: amount,
    });
  });

  it("edits only the additional scalar reset period", () => {
    const values = { ...combined, additional_max_budget: 25, additional_budget_duration: "7d" };
    expect(budgetFormPayload(values, { additional_budget_duration: true }, limits)).toEqual({
      budget_id: "both",
      budget_duration: "7d",
    });
  });

  it("does not resend the additional scalar cap when updating a calendar cap", () => {
    const values = { ...combined, annual_budget: 200, additional_max_budget: 25, additional_budget_duration: "24h" };
    expect(budgetFormPayload(values, { annual_budget: true }, limits)).toEqual({
      budget_id: "both",
      budget_limits: [limits[0], { ...limits[1], max_budget: 200 }],
    });
  });

  it.each(["30d", "12mo"])("keeps the scalar cap when changing mixed calendar limits to %s", (duration) => {
    const values = {
      ...combined,
      budget_duration: duration,
      additional_max_budget: 25,
      additional_budget_duration: "24h",
    };
    const remaining = duration === "30d" ? limits[0] : limits[1];
    expect(budgetFormPayload(values, { budget_duration: true }, limits)).toEqual({
      budget_id: "both",
      budget_limits: [{ ...remaining, max_budget: 0 }],
    });
  });

  it("adds a second calendar limit without clearing a coexisting scalar cap", () => {
    const values = { ...combined, additional_max_budget: 0, additional_budget_duration: null };
    expect(budgetFormPayload(values, { budget_duration: true, annual_budget: true }, [limits[0]])).toEqual({
      budget_id: "both",
      budget_limits: limits,
    });
  });

  it("changes a single calendar period without keeping its old reset or overwriting the scalar cap", () => {
    const values = { budget_id: "single", max_budget: 100, budget_duration: "12mo", additional_max_budget: 25 };
    expect(budgetFormPayload(values, { budget_duration: true }, [limits[0]])).toEqual({
      budget_id: "single",
      budget_limits: [{ budget_duration: "12mo", max_budget: 100 }],
    });
  });

  it("removes the original window when changing to a different scalar period", () => {
    const values = { budget_id: "single", max_budget: 100, budget_duration: "7d" };
    const expected = {
      budget_id: "single",
      max_budget: 100,
      budget_duration: "7d",
      budget_limits: null,
    };
    expect(budgetFormPayload(values, { budget_duration: true }, [limits[1]])).toEqual(expected);
  });
});
