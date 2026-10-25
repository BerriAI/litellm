import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SpendBudgetCell } from "./spend_budget_cell";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";

const indicator = (container: HTMLElement) => container.querySelector('[data-slot="meter-indicator"]');

describe("SpendBudgetCell", () => {
  it("费用与成员预算完整文案随语言切换，金额不变", () => {
    function LanguageSwitch() {
      const { setPreference } = useLanguage();
      return <button onClick={() => setPreference("en")}>English</button>;
    }
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="spend" storageScope="spend-language">
        <LanguageSwitch />
        <SpendBudgetCell spend={0} maxBudget={50} />
        <SpendBudgetCell
          spend={5}
          maxBudget={null}
          spendDecimals={2}
          teamMemberGate={{ scope: "Team member", alias: "Home", maxBudget: 100, budgetDuration: "30d", spend: 45 }}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("预算 $50")).toBeInTheDocument();
    expect(screen.getByText("· 团队成员 $45.00 / $100")).toBeInTheDocument();
    expect(screen.getAllByRole("meter")[0]).toHaveAttribute("aria-valuetext", "$0.00 / $50");
    expect(screen.getAllByRole("meter")[1]).toHaveAttribute("aria-valuenow", "45");
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByText("of $50")).toBeInTheDocument();
    expect(screen.getByText("· $45.00 of $100 (team member)")).toBeInTheDocument();
  });

  it("shows Unlimited and renders no meter when there is no budget", () => {
    const { container } = render(<SpendBudgetCell spend={0.5} maxBudget={null} />);
    expect(screen.getByText("· Unlimited")).toBeInTheDocument();
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(indicator(container)).toBeNull();
  });

  it("shows $0.00 for zero or undefined spend, never a hyphen", () => {
    const { rerender } = render(<SpendBudgetCell spend={0} maxBudget={100} />);
    expect(screen.getByText("$0.00")).toBeInTheDocument();
    expect(screen.queryByText("-")).not.toBeInTheDocument();
    rerender(<SpendBudgetCell spend={null} maxBudget={null} />);
    expect(screen.getByText("$0.00")).toBeInTheDocument();
    expect(screen.queryByText("-")).not.toBeInTheDocument();
  });

  it("renders a meter carrying the spend and budget when a budget exists", () => {
    render(<SpendBudgetCell spend={25} maxBudget={100} />);
    const meter = screen.getByRole("meter");
    expect(meter).toHaveAttribute("aria-valuenow", "25");
    expect(meter).toHaveAttribute("aria-valuemax", "100");
    expect(screen.getByText("of $100")).toBeInTheDocument();
  });

  it("supports matching spend and budget precision for summary views", () => {
    render(<SpendBudgetCell spend={98.854} maxBudget={3000} spendDecimals={2} budgetDecimals={2} />);

    expect(screen.getByText("$98.85")).toBeInTheDocument();
    expect(screen.getByText("of $3,000.00")).toBeInTheDocument();
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuetext", "$98.85 of $3,000.00");
  });

  it("keeps the default tone below 80% usage", () => {
    const { container } = render(<SpendBudgetCell spend={50} maxBudget={100} />);
    expect(indicator(container)?.className).toContain("bg-primary");
  });

  it("switches to the warning tone at 80% usage", () => {
    const { container } = render(<SpendBudgetCell spend={80} maxBudget={100} />);
    expect(indicator(container)?.className).toContain("bg-warning");
  });

  it("switches to the over tone above 100% usage", () => {
    const { container } = render(<SpendBudgetCell spend={150} maxBudget={100} />);
    expect(indicator(container)?.className).toContain("bg-destructive");
  });

  it("never meters key spend against an inherited team/org budget", () => {
    const gates = [{ scope: "Team" as const, alias: "Team A", maxBudget: 200, budgetDuration: "30d" }];
    render(<SpendBudgetCell spend={10} maxBudget={null} inheritedGates={gates} />);
    expect(screen.getByText("· Unlimited")).toBeInTheDocument();
    expect(screen.queryByText(/\(Team\)/)).not.toBeInTheDocument();
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Help")).toBeInTheDocument();
  });

  it("uses membership spend against the team-member budget when the key has no own budget", () => {
    const teamMemberGate = {
      scope: "Team member" as const,
      alias: "user@example.com in Team A",
      maxBudget: 50,
      budgetDuration: "30d",
      spend: 45,
    };
    const { container } = render(
      <SpendBudgetCell
        spend={5}
        maxBudget={null}
        teamMemberGate={teamMemberGate}
        inheritedGates={[teamMemberGate]}
        spendDecimals={2}
      />,
    );

    expect(screen.getByText("$5.00")).toBeInTheDocument();
    expect(screen.getByText("· $45.00 of $50 (team member)")).toBeInTheDocument();
    const meter = screen.getByRole("meter");
    expect(meter).toHaveAttribute("aria-valuenow", "45");
    expect(meter).toHaveAttribute("aria-valuemax", "50");
    expect(meter).toHaveAttribute("aria-valuetext", "team member spend $45.00 of $50");
    expect(indicator(container)?.className).toContain("bg-warning");
    expect(screen.getByLabelText("Help")).toBeInTheDocument();
  });

  it("uses the key's own budget instead of a team-member budget", () => {
    const teamMemberGate = {
      scope: "Team member" as const,
      alias: "user@example.com in Team A",
      maxBudget: 50,
      budgetDuration: "30d",
    };
    render(<SpendBudgetCell spend={10} maxBudget={100} teamMemberGate={teamMemberGate} />);

    expect(screen.getByText("of $100")).toBeInTheDocument();
    expect(screen.queryByText(/team member/)).not.toBeInTheDocument();
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuemax", "100");
  });

  it("shows no inherited-budget hint when there is nothing to inherit", () => {
    render(<SpendBudgetCell spend={10} maxBudget={null} inheritedGates={[]} />);
    expect(screen.queryByLabelText("Help")).not.toBeInTheDocument();
  });

  it("shows no inherited-budget hint when the key has its own budget", () => {
    const gates = [{ scope: "Team" as const, alias: "Team A", maxBudget: 200, budgetDuration: null }];
    render(<SpendBudgetCell spend={10} maxBudget={50} inheritedGates={gates} />);
    expect(screen.getByText("of $50")).toBeInTheDocument();
    expect(screen.queryByLabelText("Help")).not.toBeInTheDocument();
  });
});
