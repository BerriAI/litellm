export interface TeamAppAccess {
  team_info?: { object_permission?: { apps?: string[] | null } | null } | null;
}

export const teamAppIds = (team: TeamAppAccess | null | undefined): string[] =>
  team?.team_info?.object_permission?.apps ?? [];

export const appAccessChanged = (stored: readonly string[], selected: readonly string[]): boolean =>
  [...stored].sort().join("\n") !== [...selected].sort().join("\n");
