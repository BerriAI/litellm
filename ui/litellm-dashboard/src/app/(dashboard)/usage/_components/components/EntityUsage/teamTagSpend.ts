export const singleTeamId = (selectedTeamIds: readonly string[]): string | null =>
  selectedTeamIds.length === 1 ? selectedTeamIds[0] : null;
