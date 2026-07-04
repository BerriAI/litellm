export interface CacheDataItem {
  api_key?: string;
  model?: string;
  cache_hit_true_rows?: number | null;
  cached_completion_tokens?: number | null;
  total_rows?: number | null;
  generated_completion_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  cache_read_input_token_rows?: number | null;
  cache_activity_rows?: number | null;
  call_type?: string | null;
}

export interface CacheChartData {
  [key: string]: string | number;
  name: string;
  "LLM API requests": number;
  "Cache hit": number;
  "Cached Completion Tokens": number;
  "Provider Prompt Cache Tokens": number;
  "Generated Completion Tokens": number;
}

export interface CacheDashboardMetrics {
  chartData: CacheChartData[];
  llmApiRequests: number;
  cacheHits: number;
  cachedTokens: number;
  cacheHitRatio: string;
}

const numberOrZero = (value: number | null | undefined) => value ?? 0;

export const buildCacheDashboardMetrics = (data: CacheDataItem[]): CacheDashboardMetrics => {
  let llmApiRequests = 0;
  let cacheHits = 0;
  let cachedTokens = 0;

  const chartData = data.reduce((acc: CacheChartData[], item) => {
    const callType = item.call_type || "Unknown";
    const totalRows = numberOrZero(item.total_rows);
    const responseCacheHitRows = numberOrZero(item.cache_hit_true_rows);
    const promptCacheHitRows = numberOrZero(item.cache_read_input_token_rows);
    const cacheHitRows = item.cache_activity_rows ?? responseCacheHitRows + promptCacheHitRows;
    const cachedCompletionTokens = numberOrZero(item.cached_completion_tokens);
    const providerPromptCacheTokens = numberOrZero(item.cache_read_input_tokens);
    const generatedCompletionTokens = numberOrZero(item.generated_completion_tokens);
    const nonCacheRows = Math.max(totalRows - cacheHitRows, 0);

    llmApiRequests += nonCacheRows;
    cacheHits += cacheHitRows;
    cachedTokens += cachedCompletionTokens + providerPromptCacheTokens;

    const existingItem = acc.find((i) => i.name === callType);
    if (existingItem) {
      existingItem["LLM API requests"] += nonCacheRows;
      existingItem["Cache hit"] += cacheHitRows;
      existingItem["Cached Completion Tokens"] += cachedCompletionTokens;
      existingItem["Provider Prompt Cache Tokens"] += providerPromptCacheTokens;
      existingItem["Generated Completion Tokens"] += generatedCompletionTokens;
      return acc;
    }

    const chartItem: CacheChartData = {
      name: callType,
      "LLM API requests": nonCacheRows,
      "Cache hit": cacheHitRows,
      "Cached Completion Tokens": cachedCompletionTokens,
      "Provider Prompt Cache Tokens": providerPromptCacheTokens,
      "Generated Completion Tokens": generatedCompletionTokens,
    };

    acc.push(chartItem);
    return acc;
  }, []);

  const allRequests = cacheHits + llmApiRequests;

  return {
    chartData,
    llmApiRequests,
    cacheHits,
    cachedTokens,
    cacheHitRatio: allRequests > 0 ? ((cacheHits / allRequests) * 100).toFixed(2) : "0",
  };
};
