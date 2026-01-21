import { Team } from "@/components/key_team_helpers/key_list";

/**
 * Creates a map from team_id to team_alias for efficient lookups.
 * @param teams - Array of Team objects
 * @returns Record mapping team_id to team_alias
 */
export const createTeamAliasMap = (teams: Team[] | null | undefined): Record<string, string> => {
  if (!teams) return {};
  return teams.reduce(
    (acc, team) => {
      acc[team.team_id] = team.team_alias;
      return acc;
    },
    {} as Record<string, string>,
  );
};

/**
 * Resolves a team alias from a team ID.
 * @param teamID - The team ID to look up
 * @param teams - Array of Team objects
 * @returns The team alias if found, null otherwise
 */
export const resolveTeamAliasFromTeamID = (teamID: string, teams: Team[]): string | null => {
  const team = teams.find((team) => team.team_id === teamID);
  return team ? team.team_alias : null;
};
