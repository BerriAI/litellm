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
    expect(screen.getByText("Member")).toBeInTheDocument();
    expect(screen.getByText("$50.00")).toBeInTheDocument();
    expect(screen.getByText("/ $100.00")).toBeInTheDocument();
    expect(screen.getByText("Resets Oct 20, 2026")).toBeInTheDocument();
    expect(screen.getAllByRole("meter")[0]).toHaveAttribute("aria-valuetext", "Team $600.00 of $1,000.00");
    expect(screen.getAllByRole("meter")[1]).toHaveAttribute("aria-valuetext", "Member $50.00 of $100.00");
  });

  it("shows the team budget and Unlimited for a member without a budget", () => {
    const { container } = render(
      <TeamMemberSpendBudgetCell
        teamSpend={600}
        teamMaxBudget={1000}
        callerMembership={{ spend: 50, max_budget: null, budget_reset_at: null }}
      />,
    );

    expect(screen.getByText("Team")).toBeInTheDocument();
    expect(screen.getByText("$600.00")).toBeInTheDocument();
    expect(screen.getByText("Member")).toBeInTheDocument();
    expect(screen.getByText("/ Unlimited")).toBeInTheDocument();
    expect(screen.getAllByRole("meter")).toHaveLength(1);
    expect(container.querySelectorAll('[data-slot="meter-indicator"]')).toHaveLength(1);
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
    expect(screen.queryByText("Member")).not.toBeInTheDocument();
    expect(screen.getAllByRole("meter")).toHaveLength(1);
  });

  it("uses the over tone when member spend exceeds the budget", () => {
    const { container } = render(
      <TeamMemberSpendBudgetCell
        teamSpend={10}
        teamMaxBudget={100}
        callerMembership={{ spend: 150, max_budget: 100, budget_reset_at: null }}
      />,
    );
    const indicators = container.querySelectorAll('[data-slot="meter-indicator"]');

    expect(indicators[1]).toHaveClass("bg-destructive");
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
