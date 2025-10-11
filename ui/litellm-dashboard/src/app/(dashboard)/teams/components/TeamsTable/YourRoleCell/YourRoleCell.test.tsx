import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import type { Team } from "@/components/key_team_helpers/key_list";
import YourRoleCell from "./YourRoleCell";

// Lightweight mocks for stable, focused tests
vi.mock("@tremor/react", () => ({
  TableCell: ({ children }: { children: React.ReactNode }) => <div data-testid="cell">{children}</div>,
}));

// The component invokes TeamRoleBadge as a function, so mock it as such
vi.mock("@/app/(dashboard)/teams/components/TeamsTable/YourRoleCell/TeamRoleBadge", () => ({
  __esModule: true,
  default: (role: string | null) => <span data-testid="badge">{role === "admin" ? "Admin" : "Member"}</span>,
}));

const team = (members?: Array<{ user_id: string; role: "admin" | "user" }>): Team =>
  ({ members_with_roles: members }) as unknown as Team;

describe("YourRoleCell", () => {
  it("renders Admin when the user is an admin of the team", () => {
    render(<YourRoleCell team={team([{ user_id: "u1", role: "admin" }])} userId="u1" />);
    expect(screen.getByTestId("cell")).toBeInTheDocument();
    expect(screen.getByTestId("badge")).toHaveTextContent("Admin");
  });

  it("renders Member when the user is a regular member", () => {
    render(<YourRoleCell team={team([{ user_id: "u2", role: "user" }])} userId="u2" />);
    expect(screen.getByTestId("badge")).toHaveTextContent("Member");
  });

  it.each<[string, Team, string | null]>([
    ["userId is null", team([{ user_id: "u3", role: "admin" }]), null],
    ["user not in team", team([{ user_id: "x", role: "user" }]), "y"],
    ["team has no members", team([]), "u4"],
    ["members field undefined", team(undefined), "u5"],
  ])("falls back to Member when no role can be determined (%s)", (_label, t, uid) => {
    render(<YourRoleCell team={t} userId={uid} />);
    expect(screen.getByTestId("badge")).toHaveTextContent("Member");
  });
});
