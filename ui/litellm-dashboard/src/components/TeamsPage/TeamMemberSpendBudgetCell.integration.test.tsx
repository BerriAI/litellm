import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { TeamMemberSpendBudgetCell } from "./TeamMemberSpendBudgetCell";
import { LanguageProvider } from "@/contexts/LanguageContext";

describe("TeamMemberSpendBudgetCell", () => {
  it("随语言切换汉化预算与无障碍说明，不改变金额和 UTC 重置日期", () => {
    const content = (
      <TeamMemberSpendBudgetCell
        teamSpend={600}
        teamMaxBudget={1000}
        callerMembership={{ spend: 50, max_budget: 100, budget_reset_at: "2026-10-20T00:00:00Z" }}
      />
    );
    const { rerender } = render(
      <LanguageProvider defaultLanguage="en" userId="budget-i18n" storageScope="budget-i18n">
        {content}
      </LanguageProvider>,
    );
    expect(screen.getByText("Team")).toBeVisible();
    rerender(
      <LanguageProvider defaultLanguage="zh-CN" userId="budget-i18n" storageScope="budget-i18n">
        {content}
      </LanguageProvider>,
    );
    expect(screen.getByText("团队")).toBeVisible();
    expect(screen.getByText("我的费用")).toBeVisible();
    expect(screen.getByText("重置时间 Oct 20, 2026 UTC")).toBeVisible();
    expect(screen.getAllByRole("meter")[0]).toHaveAttribute("aria-valuetext", "团队 $600.00，总预算 $1,000.00");
    expect(screen.getAllByRole("meter")[1]).toHaveAttribute("aria-valuetext", "我的费用 $50.00，总预算 $100.00");
  });

  it("汉化无预算上限而不为零预算添加仪表", () => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="budget-unlimited" storageScope="budget-i18n">
        <TeamMemberSpendBudgetCell teamSpend={0} teamMaxBudget={0} callerMembership={{ spend: 10, max_budget: null }} />
      </LanguageProvider>,
    );
    expect(screen.getByText("/ 无限制")).toBeVisible();
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  });

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
