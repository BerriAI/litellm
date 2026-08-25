import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { InheritedBudgetHint, inheritedBudgetGates } from "./InheritedBudgetHint";

const team = { team_id: "team-1", team_alias: "Platform", max_budget: 1200, budget_duration: "30d" };
const organization = {
  organization_id: "org-1",
  organization_alias: "Acme",
  litellm_budget_table: { max_budget: 5000, budget_duration: null },
};
const user = {
  user_id: "user-1",
  user_email: "owner@example.com",
  user_alias: "Key Owner",
  max_budget: 1500,
  budget_duration: "1mo",
};

describe("inheritedBudgetGates", () => {
  it("returns team then org gates when both have budgets", () => {
    expect(inheritedBudgetGates(team, organization)).toEqual([
      { scope: "Team", alias: "Platform", maxBudget: 1200, budgetDuration: "30d" },
      { scope: "Organization", alias: "Acme", maxBudget: 5000, budgetDuration: null },
    ]);
  });

  it("skips a team or org whose max_budget is null", () => {
    expect(inheritedBudgetGates({ ...team, max_budget: null }, organization)).toEqual([
      { scope: "Organization", alias: "Acme", maxBudget: 5000, budgetDuration: null },
    ]);
    expect(inheritedBudgetGates(team, { ...organization, litellm_budget_table: { max_budget: null } })).toEqual([
      { scope: "Team", alias: "Platform", maxBudget: 1200, budgetDuration: "30d" },
    ]);
  });

  it("returns nothing when team and org are missing or budgetless", () => {
    expect(inheritedBudgetGates(null, undefined)).toEqual([]);
    expect(
      inheritedBudgetGates({ ...team, max_budget: null }, { ...organization, litellm_budget_table: null }),
    ).toEqual([]);
  });

  it("falls back to ids when aliases are empty", () => {
    expect(
      inheritedBudgetGates({ ...team, team_alias: "" }, { ...organization, organization_alias: "" }).map(
        (g) => g.alias,
      ),
    ).toEqual(["team-1", "org-1"]);
  });

  it("returns the owner's user budget as a gate", () => {
    expect(inheritedBudgetGates(null, null, user)).toEqual([
      { scope: "User", alias: "Key Owner", maxBudget: 1500, budgetDuration: "1mo" },
    ]);
  });

  it("skips the user gate when the owner has no budget", () => {
    expect(inheritedBudgetGates(null, null, { ...user, max_budget: null })).toEqual([]);
    expect(inheritedBudgetGates(null, null, null)).toEqual([]);
  });

  it("falls back to email then id for the user alias", () => {
    expect(inheritedBudgetGates(null, null, { ...user, user_alias: null })[0].alias).toBe("owner@example.com");
    expect(inheritedBudgetGates(null, null, { ...user, user_alias: null, user_email: null })[0].alias).toBe("user-1");
  });

  it("lists team, org, and user gates together", () => {
    expect(inheritedBudgetGates(team, organization, user).map((g) => g.scope)).toEqual([
      "Team",
      "Organization",
      "User",
    ]);
  });
});

describe("InheritedBudgetHint", () => {
  it("renders nothing without gates", () => {
    const { container } = render(<InheritedBudgetHint gates={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows each gate with its budget and duration on hover", async () => {
    render(<InheritedBudgetHint gates={inheritedBudgetGates(team, organization)} />);
    await userEvent.setup().hover(screen.getByLabelText("question-circle"));
    expect(screen.getByTestId("inherited-budget-hint")).toHaveTextContent("Team Platform: $1,200.00 / 30d");
    expect(screen.getByTestId("inherited-budget-hint")).toHaveTextContent("Organization Acme: $5,000.00");
    expect(screen.getByTestId("inherited-budget-hint")).not.toHaveTextContent("Organization Acme: $5,000.00 /");
  });

  it("shows the owner's user budget on hover", async () => {
    render(<InheritedBudgetHint gates={inheritedBudgetGates(null, null, user)} />);
    await userEvent.setup().hover(screen.getByLabelText("question-circle"));
    expect(screen.getByTestId("inherited-budget-hint")).toHaveTextContent("User Key Owner: $1,500.00 / 1mo");
  });
});
