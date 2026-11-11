import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  InheritedBudgetHint,
  inheritedBudgetGates,
  keyOwnerBudgetSource,
  teamMemberBudgetGate,
} from "./InheritedBudgetHint";

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

  it("places the team-member gate after the team and before organization and user gates", () => {
    const memberGate = {
      scope: "Team member" as const,
      alias: "owner@example.com in Platform",
      maxBudget: 50,
      budgetDuration: "30d",
    };

    expect(inheritedBudgetGates(team, organization, user, memberGate).map((gate) => gate.scope)).toEqual([
      "Team",
      "Team member",
      "Organization",
      "User",
    ]);
  });
});

describe("teamMemberBudgetGate", () => {
  const teamInfo = {
    team_info: {
      team_id: "team-1",
      team_alias: "Platform",
      team_member_budget_table: { max_budget: 50, budget_duration: "30d" },
    },
    team_memberships: [
      { user_id: "custom-user", spend: 14, litellm_budget_table: { max_budget: 25, budget_duration: "7d" } },
      { user_id: "default-user", spend: 35, litellm_budget_table: { max_budget: null, budget_duration: null } },
    ],
  };

  it("uses a custom membership budget instead of the team default", () => {
    const expectedGate = {
      scope: "Team member",
      alias: "custom@example.com in Platform",
      maxBudget: 25,
      budgetDuration: "7d",
      spend: 14,
    };

    expect(teamMemberBudgetGate(teamInfo, "custom-user", "custom@example.com")).toEqual(expectedGate);
  });

  it("falls back to the team default when the membership budget is null", () => {
    const expectedGate = {
      scope: "Team member",
      alias: "default-user in Platform",
      maxBudget: 50,
      budgetDuration: "30d",
      spend: 35,
    };

    expect(teamMemberBudgetGate(teamInfo, "default-user")).toEqual(expectedGate);
  });

  it("uses the team default when there is no membership row", () => {
    expect(teamMemberBudgetGate(teamInfo, "other-user")?.maxBudget).toBe(50);
  });

  it("defaults member spend to zero when there is no membership row", () => {
    const expectedGate = {
      scope: "Team member",
      alias: "other-user in Platform",
      maxBudget: 50,
      budgetDuration: "30d",
      spend: 0,
    };

    expect(teamMemberBudgetGate(teamInfo, "other-user")).toEqual(expectedGate);
  });

  it("adds an active membership increase to a custom membership budget", () => {
    const now = new Date("2026-06-20T12:00:00.000Z");
    const increasedTeamInfo = {
      ...teamInfo,
      team_memberships: [
        {
          user_id: "custom-user",
          spend: 45,
          litellm_budget_table: {
            max_budget: 25,
            budget_duration: "7d",
            temp_budget_increase: 10,
            temp_budget_expiry: "2026-06-20T12:30:00Z",
          },
        },
      ],
    };
    const expectedGate = {
      scope: "Team member",
      alias: "custom-user in Platform",
      maxBudget: 35,
      budgetDuration: "7d",
      spend: 45,
    };

    expect(teamMemberBudgetGate(increasedTeamInfo, "custom-user", null, now)).toEqual(expectedGate);
  });

  it("adds an active membership increase to the team default when the membership budget is null", () => {
    const now = new Date("2026-06-20T12:00:00.000Z");
    const increasedTeamInfo = {
      ...teamInfo,
      team_memberships: [
        {
          user_id: "default-user",
          spend: 42,
          litellm_budget_table: {
            max_budget: null,
            budget_duration: null,
            temp_budget_increase: 10,
            temp_budget_expiry: "2026-06-20T12:30:00Z",
          },
        },
      ],
    };
    const expectedGate = {
      scope: "Team member",
      alias: "default-user in Platform",
      maxBudget: 60,
      budgetDuration: "30d",
      spend: 42,
    };

    expect(teamMemberBudgetGate(increasedTeamInfo, "default-user", null, now)).toEqual(expectedGate);
  });

  it("ignores an expired membership increase", () => {
    const now = new Date("2026-06-20T12:00:00.000Z");
    const expiredTeamInfo = {
      ...teamInfo,
      team_memberships: [
        {
          user_id: "custom-user",
          litellm_budget_table: {
            max_budget: 25,
            budget_duration: "7d",
            temp_budget_increase: 10,
            temp_budget_expiry: "2026-06-20T11:30:00Z",
          },
        },
      ],
    };

    expect(teamMemberBudgetGate(expiredTeamInfo, "custom-user", null, now)?.maxBudget).toBe(25);
  });

  it("treats a naive membership expiry as UTC", () => {
    const teamInfoWithNaiveExpiry = {
      ...teamInfo,
      team_memberships: [
        {
          user_id: "default-user",
          litellm_budget_table: {
            max_budget: null,
            temp_budget_increase: 10,
            temp_budget_expiry: "2026-06-20T12:00:00",
          },
        },
      ],
    };
    const beforeExpiry = new Date("2026-06-20T11:59:59.000Z");
    const afterExpiry = new Date("2026-06-20T12:00:01.000Z");

    expect(teamMemberBudgetGate(teamInfoWithNaiveExpiry, "default-user", null, beforeExpiry)?.maxBudget).toBe(60);
    expect(teamMemberBudgetGate(teamInfoWithNaiveExpiry, "default-user", null, afterExpiry)?.maxBudget).toBe(50);
  });

  it("returns no gate when the team default is zero and the member has no budget", () => {
    expect(
      teamMemberBudgetGate(
        { ...teamInfo, team_info: { ...teamInfo.team_info, team_member_budget_table: { max_budget: 0 } } },
        "other-user",
      ),
    ).toBeNull();
  });

  it("does not create a gate from an increase when the team default is zero", () => {
    const now = new Date("2026-06-20T12:00:00.000Z");
    const noBudgetWithIncrease = {
      ...teamInfo,
      team_info: { ...teamInfo.team_info, team_member_budget_table: { max_budget: 0 } },
      team_memberships: [
        {
          user_id: "default-user",
          litellm_budget_table: {
            max_budget: null,
            temp_budget_increase: 10,
            temp_budget_expiry: "2026-06-20T12:30:00Z",
          },
        },
      ],
    };

    expect(teamMemberBudgetGate(noBudgetWithIncrease, "default-user", null, now)).toBeNull();
  });

  it("keeps an explicit zero membership budget as a gate", () => {
    const expectedGate = {
      scope: "Team member",
      alias: "zero-user in Platform",
      maxBudget: 0,
      budgetDuration: null,
      spend: 0,
    };

    expect(
      teamMemberBudgetGate(
        {
          ...teamInfo,
          team_memberships: [{ user_id: "zero-user", litellm_budget_table: { max_budget: 0 } }],
        },
        "zero-user",
      ),
    ).toEqual(expectedGate);
  });

  it("returns no gate without a user id", () => {
    expect(teamMemberBudgetGate(teamInfo, null)).toBeNull();
  });

  it("falls back to the user id and team id when labels are absent", () => {
    expect(
      teamMemberBudgetGate(
        {
          team_info: { team_id: "team-id", team_member_budget_table: { max_budget: 50 } },
        },
        "user-id",
      )?.alias,
    ).toBe("user-id in team-id");
  });
});

describe("keyOwnerBudgetSource", () => {
  it("returns the owner on a personal key regardless of the flag", () => {
    expect(keyOwnerBudgetSource({ team_id: null, user }, false)).toBe(user);
    expect(keyOwnerBudgetSource({ team_id: null, user }, true)).toBe(user);
  });

  it("hides the owner on a team key when the flag is off", () => {
    expect(keyOwnerBudgetSource({ team_id: "team-1", user }, false)).toBeNull();
  });

  it("returns the owner on a team key when the flag is on", () => {
    expect(keyOwnerBudgetSource({ team_id: "team-1", user }, true)).toBe(user);
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
