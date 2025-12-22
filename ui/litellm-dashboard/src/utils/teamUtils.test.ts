import { describe, expect, it } from "vitest";
import { resolveTeamAliasFromTeamID } from "./teamUtils";
import type { Team } from "@/components/networking";

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
