import { describe, expect, it } from "vitest";
import { resolveTeamAliasFromTeamID, createTeamAliasMap, isLikelyTeamId } from "./teamUtils";
import type { Team } from "@/components/key_team_helpers/key_list";

describe("resolveTeamAliasFromTeamID", () => {
  it("should return team alias when team is found", () => {
    const teams = [
      {
        team_id: "team1",
        team_alias: "Team One",
      },
      {
        team_id: "team2",
        team_alias: "Team Two",
      },
    ] as unknown as Team[];

    const result = resolveTeamAliasFromTeamID("team1", teams);
    expect(result).toBe("Team One");
  });

  it("should return null when team is not found", () => {
    const teams = [
      {
        team_id: "team1",
        team_alias: "Team One",
      },
      {
        team_id: "team2",
        team_alias: "Team Two",
      },
    ] as unknown as Team[];

    const result = resolveTeamAliasFromTeamID("team3", teams);
    expect(result).toBeNull();
  });
});

describe("createTeamAliasMap", () => {
  it("should create a map from team_id to team_alias", () => {
    const teams = [
      {
        team_id: "team1",
        team_alias: "Team One",
      },
      {
        team_id: "team2",
        team_alias: "Team Two",
      },
    ] as unknown as Team[];

    const result = createTeamAliasMap(teams);
    expect(result).toEqual({
      team1: "Team One",
      team2: "Team Two",
    });
  });

  it("should return empty object when teams is null or undefined", () => {
    expect(createTeamAliasMap(null)).toEqual({});
    expect(createTeamAliasMap(undefined)).toEqual({});
    expect(createTeamAliasMap([])).toEqual({});
  });
});

describe("isLikelyTeamId", () => {
  it("matches a full UUID", () => {
    expect(isLikelyTeamId("a1b2c3d4-5678-90ab-cdef-1234567890ab")).toBe(true);
  });

  it("does not match a UUID prefix or arbitrary string", () => {
    expect(isLikelyTeamId("a1b2c3d4-5678")).toBe(false);
    expect(isLikelyTeamId("my-team")).toBe(false);
  });
});
