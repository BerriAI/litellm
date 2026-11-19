import { describe, expect, it } from "vitest";
import { renderWithProviders as render, screen } from "../../../../../tests/test-utils";
import AppBudgetLimits from "./AppBudgetLimits";

const monthly = (spend: number, max_budget = 100) => [
  { source: "window" as const, budget_duration: "1mo", spend, max_budget, reset_at: null },
];

describe("concise app budget usage", () => {
  it("shows rate-only limits without describing the app as unlimited", () => {
    render(<AppBudgetLimits budget={{ tpm_limit: 50000, rpm_limit: 0 }} />);
    expect(screen.getByText("50,000 TPM")).toBeInTheDocument();
    expect(screen.getByText("0 RPM")).toBeInTheDocument();
    expect(screen.getByText("No spend limit")).toBeInTheDocument();
    expect(screen.queryByText("Unlimited")).not.toBeInTheDocument();
  });

  it("should show a zero spend against its allowance once", () => {
    render(<AppBudgetLimits usage={monthly(0)} cumulativeSpend={500} />);
    expect(screen.getByText("$0.00 of $100.00 used")).toBeInTheDocument();
    expect(screen.getByText("$100.00 remaining")).toBeInTheDocument();
    expect(screen.queryByText("$500.00")).not.toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "App budget used" })).toHaveAttribute("aria-valuenow", "0");
  });
  it("should show tiny positive usage without eight-decimal figures or a false zero", () => {
    render(<AppBudgetLimits usage={monthly(0.0005481)} />);
    expect(screen.getByText("<$0.01 of $100.00 used")).toBeInTheDocument();
    expect(screen.queryByText("$0.00 of $100.00 used")).not.toBeInTheDocument();
    expect(screen.queryByText("$0.0005481")).not.toBeInTheDocument();
  });
  it("should preserve a tiny positive remaining balance", () => {
    render(<AppBudgetLimits usage={monthly(0.9999, 1)} />);
    expect(screen.getByText("<$0.01 remaining")).toBeInTheDocument();
    expect(screen.queryByText("Budget exhausted")).not.toBeInTheDocument();
  });
  it.each([
    [0, 0],
    [0.0001, 0],
    [100, 100],
  ])("should label exhausted usage %s against %s clearly", (spend, limit) => {
    render(<AppBudgetLimits usage={monthly(spend, limit)} />);
    expect(screen.getByText("Budget exhausted")).toBeInTheDocument();
    expect(screen.queryByText("$0.00 remaining")).not.toBeInTheDocument();
  });
  it("should display a profile allowance in standard currency", () => {
    render(<AppBudgetLimits budget={{ max_budget: 100, budget_duration: "1mo" }} />);
    expect(screen.getByText("$100.00")).toBeInTheDocument();
    expect(screen.getByText("Monthly")).toBeInTheDocument();
    expect(screen.queryByText("Period usage unavailable")).not.toBeInTheDocument();
  });
});
