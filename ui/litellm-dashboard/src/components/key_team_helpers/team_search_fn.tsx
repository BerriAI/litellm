import { Team } from "./key_list";

export const createTeamSearchFunction = (teams: Team[] | null) => {
  return async (searchText: string): Promise<Array<{ label: string; value: string }>> => {
    // Return empty array if teams is null or searchText is empty
    if (!teams || !searchText.trim()) {
      return [];
    }

    // Filter teams where team_alias contains the search text (case insensitive)
    const filteredTeams = teams.filter((team) => team.team_alias.toLowerCase().includes(searchText.toLowerCase()));

    // Map filtered teams to the required format
    return filteredTeams.map((team) => ({
      label: `${team.team_alias} (${team.team_id.substring(0, 8)}...)`,
      value: team.team_id,
    }));
  };
};
