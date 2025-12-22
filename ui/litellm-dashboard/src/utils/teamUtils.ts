import { Team } from "@/components/networking";

export const resolveTeamAliasFromTeamID = (teamID: string, teams: Team[]): string | null => {
  const team = teams.find((team) => team.team_id === teamID);
  return team ? team.team_alias : null;
};
