// Mirrors request-time matching (RouteChecks._route_matches_wildcard_pattern): only a
// trailing "*" is a wildcard (prefix match). Anything else - including a "?" or a
// non-trailing "*" - is compared by exact equality when a request is matched, so it is
// treated as a concrete alias that must exist.
export const isWildcardPattern = (value: string): boolean => value.endsWith("*");

export const getInvalidTeamEntries = (values: string[], availableTeams: string[]): string[] =>
  values.filter((value) => !isWildcardPattern(value) && !availableTeams.includes(value));
