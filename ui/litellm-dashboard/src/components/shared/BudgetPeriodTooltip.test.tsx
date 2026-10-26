import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import { BudgetPeriodTooltip, budgetPeriodLines } from "./BudgetPeriodTooltip";

const originalTimezone: string | undefined = process.env.TZ;

afterEach(() => {
  if (originalTimezone === undefined) {
    delete process.env.TZ;
    return;
  }
  process.env.TZ = originalTimezone;
});

const scheduled = { spend: 12.5, maxBudget: 100, budgetDuration: "30d", budgetResetAt: "2026-04-01T00:00:00Z" };
const unparseableDuration = { ...scheduled, spend: 0, budgetDuration: "fortnight" };
const unscheduled = { spend: 0, maxBudget: 100, budgetDuration: "30d", budgetResetAt: null };
const lifetime = { spend: 3, maxBudget: 50, budgetDuration: null, budgetResetAt: null };
const noBudget = { spend: null, maxBudget: null, budgetDuration: "30d", budgetResetAt: null };
const noBudgetNoDuration = { spend: 2, maxBudget: null, budgetDuration: null, budgetResetAt: null };

describe("budgetPeriodLines", () => {
  it("describes the current period, max budget, interval and spend when a reset is scheduled", () => {
    process.env.TZ = "UTC";
    expect(budgetPeriodLines(scheduled)).toEqual([
      "Budget period: Mar 1, 2026 00:00 to Apr 1, 2026 00:00 (UTC)",
      "Max budget: $100.00",
      "Reset interval: monthly",
      "Spend this period: $12.5000",
    ]);
  });

  it("falls back to the next reset when the duration cannot be parsed", () => {
    process.env.TZ = "UTC";
    expect(budgetPeriodLines(unparseableDuration)).toEqual([
      "Next reset: Apr 1, 2026 00:00 (UTC)",
      "Max budget: $100.00",
      "Reset interval: fortnight",
      "Spend this period: $0.0000",
    ]);
  });

  it("omits the period line when no reset has been scheduled yet", () => {
    expect(budgetPeriodLines(unscheduled)).toEqual([
      "Max budget: $100.00",
      "Reset interval: monthly",
      "Spend this period: $0.0000",
    ]);
  });

  it("reports a lifetime budget when there is no duration", () => {
    expect(budgetPeriodLines(lifetime)).toEqual([
      "Lifetime budget, never resets",
      "Max budget: $50.00",
      "Spend so far: $3.0000",
    ]);
  });

  it("reports no budget when max budget is unset", () => {
    expect(budgetPeriodLines(noBudget)).toEqual(["No budget set", "Spend this period: $0.0000"]);
  });

  it("uses lifetime wording when there is neither a budget nor a duration", () => {
    expect(budgetPeriodLines(noBudgetNoDuration)).toEqual(["No budget set", "Spend so far: $2.0000"]);
  });
});

describe("BudgetPeriodTooltip", () => {
  it("shows the budget period details on hover", async () => {
    const user = userEvent.setup();
    render(<BudgetPeriodTooltip spend={1} maxBudget={10} budgetDuration="1d" budgetResetAt="2026-10-27T00:00:00Z" />);

    await user.hover(screen.getByLabelText("budget-period-details"));

    const tooltip = await screen.findByTestId("budget-period-tooltip");
    expect(tooltip).toHaveTextContent("Max budget: $10.00");
    expect(tooltip).toHaveTextContent("Reset interval: 1d");
    expect(tooltip).toHaveTextContent("Spend this period: $1.0000");
    expect(tooltip).not.toHaveTextContent("Invalid Date");
  });
});
