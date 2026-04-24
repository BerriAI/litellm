import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { Team } from "@/components/key_team_helpers/key_list";
import TeamsTable from "./TeamsTable";

vi.mock("@/components/ui/table", () => ({
  Table: ({ children }: any) => <table>{children}</table>,
  TableHeader: ({ children }: any) => <thead>{children}</thead>,
  TableBody: ({ children }: any) => <tbody>{children}</tbody>,
  TableRow: ({ children }: any) => <tr>{children}</tr>,
  TableHead: ({ children }: any) => <th>{children}</th>,
  TableCell: ({ children, ...props }: any) => <td {...props}>{children}</td>,
}));

vi.mock("@/utils/dataUtils", () => ({
  formatNumberWithCommas: (val: number, decimals: number) =>
    val != null ? val.toFixed(decimals) : "N/A",
}));

vi.mock("@/app/(dashboard)/teams/components/TeamsTable/ModelsCell", () => ({
  default: ({ team }: any) => <td data-testid="models-cell">{team.models.join(",")}</td>,
}));

vi.mock("@/app/(dashboard)/teams/components/TeamsTable/YourRoleCell/YourRoleCell", () => ({
  default: ({ team }: any) => <td data-testid="role-cell">{team.team_id}</td>,
}));

const makeTeam = (overrides: Partial<Team> = {}): Team => ({
  team_id: "team-abc1234",
  team_alias: "Platform",
  models: ["gpt-4"],
  max_budget: 500,
  budget_duration: null,
  tpm_limit: null,
  rpm_limit: null,
  organization_id: "org-1",
  created_at: "2024-06-01T00:00:00Z",
  keys: [],
  members_with_roles: [],
  spend: 123.4567,
  ...overrides,
});

const defaultPerTeamInfo = {
  "team-abc1234": {
    keys: [{ token: "tok-1" } as any, { token: "tok-2" } as any],
    team_info: {
      members_with_roles: [{ user_id: "u1", role: "admin" } as any],
    },
  },
};

const renderTable = (overrides: Partial<Parameters<typeof TeamsTable>[0]> = {}) => {
  const defaults = {
    teams: [makeTeam()],
    currentOrg: null,
    perTeamInfo: defaultPerTeamInfo,
    userRole: "Admin",
    userId: "user-1",
    setSelectedTeamId: vi.fn(),
    setEditTeam: vi.fn(),
    onDeleteTeam: vi.fn(),
  };
  return render(<TeamsTable {...defaults} {...overrides} />);
};

describe("TeamsTable", () => {
  it("should render table headers", () => {
    renderTable();

    expect(screen.getByText("Team Name")).toBeInTheDocument();
    expect(screen.getByText("Team ID")).toBeInTheDocument();
    expect(screen.getByText("Created")).toBeInTheDocument();
    expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
    expect(screen.getByText("Budget (USD)")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("Organization")).toBeInTheDocument();
    expect(screen.getByText("Your Role")).toBeInTheDocument();
    expect(screen.getByText("Info")).toBeInTheDocument();
  });

  it("should render team rows with team data", () => {
    renderTable();

    expect(screen.getByText("Platform")).toBeInTheDocument();
    expect(screen.getByText("team-ab...")).toBeInTheDocument();
    expect(screen.getByText("org-1")).toBeInTheDocument();
  });

  it("should show edit and delete buttons for Admin users", () => {
    renderTable({ userRole: "Admin" });

    expect(
      screen.getByRole("button", { name: /edit team/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /delete team/i }),
    ).toBeInTheDocument();
  });

  it("should not show edit and delete buttons for non-Admin users", () => {
    renderTable({ userRole: "Internal User" });

    expect(
      screen.queryByRole("button", { name: /edit team/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /delete team/i }),
    ).not.toBeInTheDocument();
  });

  it("should call setSelectedTeamId when team ID button is clicked", async () => {
    const user = userEvent.setup();
    const setSelectedTeamId = vi.fn();
    renderTable({ setSelectedTeamId });

    await user.click(screen.getByTestId("team-id-cell"));

    expect(setSelectedTeamId).toHaveBeenCalledWith("team-abc1234");
  });
});
