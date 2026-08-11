const WORD_FORM_BUDGET_DURATIONS: Record<string, string> = {
  hourly: "1h",
  daily: "24h",
  weekly: "7d",
  monthly: "30d",
};

// Normalize any legacy word-form budget duration to the canonical value the dropdown uses
export const canonicalBudgetDuration = (duration: string | null | undefined): string | null =>
  duration ? WORD_FORM_BUDGET_DURATIONS[duration] ?? duration : null;

// Determine the key_type display value from allowed_routes
const KEY_TYPE_BY_PRESET_ROUTE: Record<string, string> = {
  llm_api_routes: "llm_api",
  management_routes: "management",
  info_routes: "read_only",
};

export const exactKeyTypePresetFromRoutes = (allowedRoutes: string[] | null | undefined): string | undefined => {
  if (!allowedRoutes || allowedRoutes.length === 0) return "default";
  if (allowedRoutes.length !== 1) return undefined;
  return KEY_TYPE_BY_PRESET_ROUTE[allowedRoutes[0]];
};

// Determine the key_type display value from allowed_routes
export const keyTypeFromRoutes = (allowedRoutes: string[] | null | undefined): string =>
  exactKeyTypePresetFromRoutes(allowedRoutes) ?? "default";

export const parseAllowedRoutes = (value: unknown): string[] =>
  typeof value === "string" && value.trim() !== ""
    ? value
        .split(",")
        .map((route) => route.trim())
        .filter((route) => route.length > 0)
    : [];

export const modelSentinelOptions = (
  keyTeamId: string | null | undefined,
  teamLoaded: boolean,
): { value: string; label: string }[] => {
  if (keyTeamId == null) return [{ value: "all-proxy-models", label: "All Proxy Models" }];
  return teamLoaded ? [{ value: "all-team-models", label: "All Team Models" }] : [];
};

export const currentValuePlaceholder = (
  premiumUser: boolean,
  current: unknown,
  premiumHint: string,
  emptyHint: string,
): string => {
  if (!premiumUser) return premiumHint;
  return Array.isArray(current) && current.length > 0 ? `Current: ${current.join(", ")}` : emptyHint;
};

