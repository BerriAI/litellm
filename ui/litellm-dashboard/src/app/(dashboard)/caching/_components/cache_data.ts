export interface CacheActivityRow {
  api_key: string;
  model: string;
  call_type: string;
  total_rows: number;
  cache_hit_true_rows: number;
  failed_rows?: number;
  cached_completion_tokens?: number;
  generated_completion_tokens?: number;
}

export const UNKNOWN_CALL_TYPE = "Unknown";

export const REQUEST_SERIES = {
  apiRequests: "LLM API requests",
  cacheHits: "Cache hit",
  failed: "Failed requests",
} as const;

export type CacheChartDatum = {
  name: string;
  [REQUEST_SERIES.apiRequests]: number;
  [REQUEST_SERIES.cacheHits]: number;
  [REQUEST_SERIES.failed]: number;
  "Cached Completion Tokens": number;
  "Generated Completion Tokens": number;
};

export interface CacheActivitySummary {
  chartData: CacheChartDatum[];
  cacheHits: number;
  llmApiRequests: number;
  failedRequests: number;
  cachedCompletionTokens: number;
}

export function summarizeCacheActivity(rows: readonly CacheActivityRow[]): CacheActivitySummary {
  const groups = new Map<string, CacheChartDatum>();

  for (const row of rows) {
    const name = row.call_type || UNKNOWN_CALL_TYPE;
    const hits = row.cache_hit_true_rows || 0;
    const failed = row.failed_rows || 0;
    const apiRequests = (row.total_rows || 0) - hits - failed;

    const group = groups.get(name) ?? {
      name,
      [REQUEST_SERIES.apiRequests]: 0,
      [REQUEST_SERIES.cacheHits]: 0,
      [REQUEST_SERIES.failed]: 0,
      "Cached Completion Tokens": 0,
      "Generated Completion Tokens": 0,
    };

    groups.set(name, {
      ...group,
      [REQUEST_SERIES.apiRequests]: group[REQUEST_SERIES.apiRequests] + apiRequests,
      [REQUEST_SERIES.cacheHits]: group[REQUEST_SERIES.cacheHits] + hits,
      [REQUEST_SERIES.failed]: group[REQUEST_SERIES.failed] + failed,
      "Cached Completion Tokens": group["Cached Completion Tokens"] + (row.cached_completion_tokens || 0),
      "Generated Completion Tokens": group["Generated Completion Tokens"] + (row.generated_completion_tokens || 0),
    });
  }

  const chartData = Array.from(groups.values());
  return {
    chartData,
    cacheHits: chartData.reduce((sum, g) => sum + g[REQUEST_SERIES.cacheHits], 0),
    llmApiRequests: chartData.reduce((sum, g) => sum + g[REQUEST_SERIES.apiRequests], 0),
    failedRequests: chartData.reduce((sum, g) => sum + g[REQUEST_SERIES.failed], 0),
    cachedCompletionTokens: chartData.reduce((sum, g) => sum + g["Cached Completion Tokens"], 0),
  };
}
