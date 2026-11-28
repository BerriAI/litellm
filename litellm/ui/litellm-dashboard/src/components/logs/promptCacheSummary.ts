import type { TFunction } from "i18next";

import { extractPromptCacheTokens, type PromptCacheTokens } from "@/utils/promptCacheUsage";

import type { LogEntry, SessionPromptCacheTotals } from "./types";

export function requestPromptCacheTokens(log: LogEntry): PromptCacheTokens {
  if (String(log.cache_hit).toLowerCase() === "true") return {};
  return extractPromptCacheTokens(log.metadata?.additional_usage_values, true);
}

export function formatPromptCacheSummary(
  t: TFunction,
  { cacheReadTokens, cacheCreationTokens }: PromptCacheTokens,
): string | undefined {
  if (cacheReadTokens === undefined && cacheCreationTokens === undefined) return undefined;
  return t("logs.request.columns.promptCacheSummary", {
    read: cacheReadTokens ?? 0,
    write: cacheCreationTokens ?? 0,
  });
}

export function formatSessionPromptCacheSummary(t: TFunction, totals: SessionPromptCacheTotals): string | undefined {
  const read = totals.session_cache_read_tokens;
  const summary = formatPromptCacheSummary(t, {
    cacheReadTokens: read ?? undefined,
    cacheCreationTokens: totals.session_cache_write_tokens ?? undefined,
  });
  const input = totals.session_provider_prompt_tokens;
  if (summary === undefined || read == null) return summary;
  if (!input || read > input) return summary;
  return t("logs.request.columns.promptCacheSessionSummary", { summary, share: read / input });
}
