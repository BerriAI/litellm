import { describe, expect, it } from "vitest";
import { buildCacheDashboardMetrics } from "./cacheDashboardMetrics";

describe("buildCacheDashboardMetrics", () => {
  it("should include provider prompt cache tokens in the cache summary", () => {
    const result = buildCacheDashboardMetrics([
      {
        api_key: "sk-test",
        call_type: "aresponses",
        model: "bedrock_mantle/openai.gpt-5.5",
        total_rows: 52,
        cache_hit_true_rows: null,
        cached_completion_tokens: null,
        generated_completion_tokens: 42,
        cache_read_input_token_rows: 42,
        cache_activity_rows: 42,
        cache_read_input_tokens: 86941,
        cache_creation_input_tokens: 0,
      },
    ]);

    expect(result.cacheHits).toBe(42);
    expect(result.cacheHitRatio).toBe("80.77");
    expect(result.cachedTokens).toBe(86941);
    expect(result.chartData[0]["Provider Prompt Cache Tokens"]).toBe(86941);
  });

  it("should not double count rows cached by both response and provider prompt caching", () => {
    const result = buildCacheDashboardMetrics([
      {
        call_type: "acompletion",
        total_rows: 10,
        cache_hit_true_rows: 5,
        cache_read_input_token_rows: 5,
        cache_activity_rows: 5,
        cached_completion_tokens: 100,
        cache_read_input_tokens: 200,
        cache_creation_input_tokens: 50,
        generated_completion_tokens: 50,
      },
    ]);

    expect(result.cacheHits).toBe(5);
    expect(result.cacheHitRatio).toBe("50.00");
    expect(result.cachedTokens).toBe(300);
    expect(result.chartData[0]["LLM API requests"]).toBe(5);
    expect(result.chartData[0]["Cache hit"]).toBe(5);
    expect(result.chartData[0]["Provider Prompt Cache Tokens"]).toBe(200);
  });
});
