import { describe, expect, it, vi } from "vitest";

import type { TeamsResponse } from "@/app/(dashboard)/hooks/teams/useTeams";

import type { Team } from "../key_team_helpers/key_list";
import {
  buildTeamsCsv,
  buildTeamsCsvRows,
  collectTeamMemberBudgetIds,
  fetchAllTeams,
  TEAMS_EXPORT_PAGE_SIZE,
} from "./teamsCsvExport";

const makeTeam = (overrides: Partial<Team>): Team =>
  ({
    team_id: "team-1",
    team_alias: "alias-1",
    models: [],
    max_budget: null,
    budget_duration: null,
    tpm_limit: null,
    rpm_limit: null,
    organization_id: "org-1",
    created_at: "2026-01-01T00:00:00Z",
    keys: [],
    members_with_roles: [],
    spend: 0,
    ...overrides,
  }) as Team;

const makePage = (teams: Team[], page: number, totalPages: number): TeamsResponse => ({
  teams,
  total: teams.length,
  page,
  page_size: TEAMS_EXPORT_PAGE_SIZE,
  total_pages: totalPages,
});

describe("fetchAllTeams", () => {
  it("returns the single page without extra requests", async () => {
    const fetchPage = vi.fn().mockResolvedValue(makePage([makeTeam({ team_id: "a" })], 1, 1));
    const teams = await fetchAllTeams(fetchPage);
    expect(teams.map((t) => t.team_id)).toEqual(["a"]);
    expect(fetchPage).toHaveBeenCalledTimes(1);
    expect(fetchPage).toHaveBeenCalledWith(1, TEAMS_EXPORT_PAGE_SIZE);
  });

  it("fetches and concatenates every page in order", async () => {
    const fetchPage = vi
      .fn()
      .mockImplementation(async (page: number) => makePage([makeTeam({ team_id: `team-${page}` })], page, 3));
    const teams = await fetchAllTeams(fetchPage);
    expect(teams.map((t) => t.team_id)).toEqual(["team-1", "team-2", "team-3"]);
    expect(fetchPage).toHaveBeenCalledTimes(3);
    expect(fetchPage).toHaveBeenCalledWith(2, TEAMS_EXPORT_PAGE_SIZE);
    expect(fetchPage).toHaveBeenCalledWith(3, TEAMS_EXPORT_PAGE_SIZE);
  });
});

describe("collectTeamMemberBudgetIds", () => {
  it("dedupes ids and skips teams without a member budget", () => {
    const teams = [
      makeTeam({ team_id: "a", metadata: { team_member_budget_id: "bud-1" } }),
      makeTeam({ team_id: "b", metadata: { team_member_budget_id: "bud-1" } }),
      makeTeam({ team_id: "c", metadata: {} }),
      makeTeam({ team_id: "d", metadata: { team_member_budget_id: "" } }),
      makeTeam({ team_id: "e" }),
      makeTeam({ team_id: "f", metadata: { team_member_budget_id: "bud-2" } }),
    ];
    expect(collectTeamMemberBudgetIds(teams)).toEqual(["bud-1", "bud-2"]);
  });
});

describe("buildTeamsCsvRows", () => {
  it("maps configured limits, spend, models, and rate limits", () => {
    const teamFields: Partial<Team> = {
      team_id: "team-42",
      team_alias: "finance",
      organization_id: "org-9",
      models: ["gpt-4o", "claude-sonnet-4-5"],
      max_budget: 250,
      budget_duration: "30d",
      budget_reset_at: "2026-02-01T00:00:00Z",
      spend: 12.5,
      tpm_limit: 1000,
      rpm_limit: 50,
      members_count: 7,
      keys_count: 3,
      blocked: false,
    };
    const [row] = buildTeamsCsvRows([makeTeam(teamFields)], []);
    const expectedRow = {
      "Team Alias": "finance",
      "Team ID": "team-42",
      "Organization ID": "org-9",
      Models: "gpt-4o, claude-sonnet-4-5",
      "Max Budget (USD)": 250,
      "Budget Duration": "30d",
      "Budget Reset At": "2026-02-01T00:00:00Z",
      "Spend (USD)": 12.5,
      "TPM Limit": 1000,
      "RPM Limit": 50,
      "Team Member Budget (USD)": "",
      "Team Member Budget Duration": "",
      "Team Member TPM Limit": "",
      "Team Member RPM Limit": "",
      Members: 7,
      Keys: 3,
      Blocked: false,
      "Created At": "2026-01-01T00:00:00Z",
    };
    expect(row).toEqual(expectedRow);
  });

  it("joins team member budget rows by budget id from metadata", () => {
    const teams = [
      makeTeam({ team_id: "a", metadata: { team_member_budget_id: "bud-1" } }),
      makeTeam({ team_id: "b" }),
    ];
    const rows = buildTeamsCsvRows(teams, [
      { budget_id: "bud-1", max_budget: 25, budget_duration: "7d", tpm_limit: 200, rpm_limit: 10 },
    ]);
    expect(rows[0]["Team Member Budget (USD)"]).toBe(25);
    expect(rows[0]["Team Member Budget Duration"]).toBe("7d");
    expect(rows[0]["Team Member TPM Limit"]).toBe(200);
    expect(rows[0]["Team Member RPM Limit"]).toBe(10);
    expect(rows[1]["Team Member Budget (USD)"]).toBe("");
  });

  it("falls back to members_with_roles and keys lengths when counts are absent", () => {
    const team = makeTeam({
      members_with_roles: [
        { user_id: "u1", role: "admin" },
        { user_id: "u2", role: "user" },
      ],
      keys: [{ token: "t" } as Team["keys"][number]],
    });
    const [row] = buildTeamsCsvRows([team], []);
    expect(row.Members).toBe(2);
    expect(row.Keys).toBe(1);
  });
});

describe("buildTeamsCsv", () => {
  it("produces a header row and quotes values containing commas", () => {
    const csv = buildTeamsCsv([makeTeam({ team_alias: "sales, emea", models: ["m1", "m2"] })], []);
    const [header, row] = csv.split("\r\n");
    expect(header).toBe(
      "Team Alias,Team ID,Organization ID,Models,Max Budget (USD),Budget Duration,Budget Reset At,Spend (USD)," +
        "TPM Limit,RPM Limit,Team Member Budget (USD),Team Member Budget Duration,Team Member TPM Limit," +
        "Team Member RPM Limit,Members,Keys,Blocked,Created At",
    );
    expect(row).toContain('"sales, emea"');
    expect(row).toContain('"m1, m2"');
  });

  it("neutralizes formula-leading values so spreadsheets render them as text", () => {
    const csv = buildTeamsCsv([makeTeam({ team_alias: "=SUM(A1:A9)" })], []);
    const [, row] = csv.split("\r\n");
    expect(row).toContain('"\'=SUM(A1:A9)"');
    expect(row).not.toContain("=SUM(A1:A9),");
  });
});
