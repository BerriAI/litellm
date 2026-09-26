export interface ApiKeyTruncation {
  limit: number;
  total: number;
}

export interface UsageFetchState {
  coversRange: boolean;
  cancelled: boolean;
  failed: boolean;
  apiKeyTruncation?: ApiKeyTruncation | null;
}

export const getApiKeyTruncation = (apiKeyLimit: unknown, totalApiKeys: unknown): ApiKeyTruncation | undefined => {
  if (typeof apiKeyLimit !== "number" || typeof totalApiKeys !== "number") return undefined;
  return totalApiKeys > apiKeyLimit ? { limit: apiKeyLimit, total: totalApiKeys } : undefined;
};

export const getExportBlockedReason = (
  { coversRange, cancelled, failed, apiKeyTruncation }: UsageFetchState,
  exportNoun: string = "a per-team export",
): string | undefined => {
  if (failed) return "Some spend data failed to load, so an export would under-report. Reload the page to try again.";
  if (cancelled)
    return "Loading was stopped before the whole range arrived, so an export would under-report. Reload the page to load it all.";
  if (!coversRange) return "Spend data is still loading, so an export would under-report. Wait for it to finish.";
  if (apiKeyTruncation)
    return `Only the ${apiKeyTruncation.limit} highest-spend keys of ${apiKeyTruncation.total} were loaded, so ${exportNoun} would under-report. Raise USAGE_TOP_API_KEYS_LIMIT on the proxy to load more keys.`;
  return undefined;
};
