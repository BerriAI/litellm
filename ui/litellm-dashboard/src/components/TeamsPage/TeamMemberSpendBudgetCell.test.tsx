import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { TeamMemberSpendBudgetCell } from "./TeamMemberSpendBudgetCell";

describe("TeamMemberSpendBudgetCell", () => {
  it("shows team and member budgets with the member reset date", () => {
    render(
      <TeamMemberSpendBudgetCell
        teamSpend={600}
        teamMaxBudget={1000}
        callerMembership={{
          spend: 50,
          max_budget: 100,
          budget_reset_at: "2026-10-20T12:00:00Z",
        }}
      />,
    );

    expect(screen.getByText("Team")).toBeInTheDocument();
    expect(screen.getByText("$600.00")).toBeInTheDocument();
    expect(screen.getByText("/ $1,000.00")).toBeInTheDocument();
    expect(screen.getByText("My Spend")).toBeInTheDocument();
    expect(screen.getByText("$50.00")).toBeInTheDocument();
    expect(screen.getByText("/ $100.00")).toBeInTheDocument();
    expect(screen.getByText("Resets Oct 20, 2026 UTC")).toBeInTheDocument();
    expect(screen.getAllByRole("meter")[0]).toHaveAttribute("aria-valuetext", "Team $600.00 of $1,000.00");
    expect(screen.getAllByRole("meter")[1]).toHaveAttribute("aria-valuetext", "My Spend $50.00 of $100.00");
  });

  it("shows the team budget and Unlimited for a member without a budget", () => {
    render(
      <TeamMemberSpendBudgetCell
        teamSpend={600}
        teamMaxBudget={1000}
        callerMembership={{ spend: 50, max_budget: null, budget_reset_at: null }}
      />,
    );

    expect(screen.getByText("Team")).toBeInTheDocument();
    expect(screen.getByText("$600.00")).toBeInTheDocument();
    expect(screen.getByText("My Spend")).toBeInTheDocument();
    expect(screen.getByText("/ Unlimited")).toBeInTheDocument();
    expect(screen.getAllByRole("meter")).toHaveLength(1);
  });

  it("omits the reset text when no member reset date exists", () => {
    render(
      <TeamMemberSpendBudgetCell
        teamSpend={600}
        teamMaxBudget={1000}
        callerMembership={{ spend: 50, max_budget: 100, budget_reset_at: null }}
      />,
    );

    expect(screen.queryByText(/Resets/)).not.toBeInTheDocument();
  });

  it("renders only the team line when caller membership is absent", () => {
    render(<TeamMemberSpendBudgetCell teamSpend={600} teamMaxBudget={1000} callerMembership={null} />);

    expect(screen.getByText("Team")).toBeInTheDocument();
    expect(screen.queryByText("My Spend")).not.toBeInTheDocument();
    expect(screen.getAllByRole("meter")).toHaveLength(1);
  });

  it("exposes member spend above its budget through the meter", () => {
    render(
      <TeamMemberSpendBudgetCell
        teamSpend={10}
        teamMaxBudget={100}
        callerMembership={{ spend: 150, max_budget: 100, budget_reset_at: null }}
      />,
    );

    expect(screen.getAllByRole("meter")[1]).toHaveAttribute("aria-valuetext", "My Spend $150.00 of $100.00");
  });

  it("does not render meters for zero or unlimited budgets", () => {
    render(
      <TeamMemberSpendBudgetCell
        teamSpend={10}
        teamMaxBudget={0}
        callerMembership={{ spend: 10, max_budget: null, budget_reset_at: null }}
      />,
    );

    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  });
});
