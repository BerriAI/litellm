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
export const keyTypeFromRoutes = (allowedRoutes: string[] | null | undefined): string => {
  if (!allowedRoutes || allowedRoutes.length === 0) return "default";
  if (allowedRoutes.includes("llm_api_routes")) return "llm_api";
  if (allowedRoutes.includes("management_routes")) return "management";
  if (allowedRoutes.includes("info_routes")) return "read_only";
  return "default";
};

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

export type MovedMetadataTags = {
  metadata: string;
  tags: string[];
  movedTags: string[];
};

const parseJsonObject = (text: string): Record<string, unknown> | null => {
  try {
    const parsed: unknown = JSON.parse(text);
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
};

export const moveTagsOutOfMetadataJson = (
  metadataJson: string | undefined,
  currentTags: readonly string[] | undefined,
): MovedMetadataTags | null => {
  const parsed = metadataJson === undefined ? null : parseJsonObject(metadataJson);
  if (parsed === null) return null;
  const { tags: metadataTags, ...rest } = parsed;
  if (!Array.isArray(metadataTags)) return null;
  const existing = currentTags ?? [];
  const movedTags = metadataTags.filter(
    (tag: unknown, index: number): tag is string =>
      typeof tag === "string" && !existing.includes(tag) && metadataTags.indexOf(tag) === index,
  );
  return { metadata: JSON.stringify(rest, null, 2), tags: [...existing, ...movedTags], movedTags };
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
