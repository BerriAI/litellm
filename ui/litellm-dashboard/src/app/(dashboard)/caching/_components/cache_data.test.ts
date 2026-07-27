import { describe, expect, it } from "vitest";
import { summarizeCacheActivity, UNKNOWN_CALL_TYPE, type CacheActivityRow } from "./cache_data";

const row = (overrides: Partial<CacheActivityRow>): CacheActivityRow => ({
  api_key: "sk-1",
  model: "gpt-5.1",
  call_type: "acompletion",
  total_rows: 0,
  cache_hit_true_rows: 0,
  ...overrides,
});

describe("summarizeCacheActivity", () => {
  it("splits each call_type into api requests, cache hits, and failed requests", () => {
    const summary = summarizeCacheActivity([row({ total_rows: 1000, cache_hit_true_rows: 300, failed_rows: 100 })]);

    expect(summary.chartData).toEqual([
      expect.objectContaining({
        name: "acompletion",
        "LLM API requests": 600,
        "Cache hit": 300,
        "Failed requests": 100,
      }),
    ]);
    expect(summary.llmApiRequests).toBe(600);
    expect(summary.cacheHits).toBe(300);
    expect(summary.failedRequests).toBe(100);
  });

  it("treats a missing failed_rows as zero so older proxy responses keep the old math", () => {
    const summary = summarizeCacheActivity([row({ total_rows: 50, cache_hit_true_rows: 20 })]);

    expect(summary.chartData[0]["LLM API requests"]).toBe(30);
    expect(summary.chartData[0]["Failed requests"]).toBe(0);
  });

  it("buckets rows with an empty call_type under Unknown", () => {
    const summary = summarizeCacheActivity([
      row({ call_type: "", total_rows: 8, failed_rows: 8 }),
      row({ call_type: "", api_key: "sk-2", total_rows: 3, failed_rows: 3 }),
    ]);

    expect(summary.chartData).toHaveLength(1);
    expect(summary.chartData[0]).toEqual(
      expect.objectContaining({ name: UNKNOWN_CALL_TYPE, "Failed requests": 11, "LLM API requests": 0 }),
    );
  });

  it("merges rows sharing a call_type across keys and models", () => {
    const summary = summarizeCacheActivity([
      row({ total_rows: 10, cache_hit_true_rows: 4, failed_rows: 1, cached_completion_tokens: 100 }),
      row({
        api_key: "sk-2",
        model: "claude-opus-4-8",
        total_rows: 20,
        cache_hit_true_rows: 6,
        failed_rows: 2,
        generated_completion_tokens: 500,
      }),
    ]);

    expect(summary.chartData).toEqual([
      {
        name: "acompletion",
        "LLM API requests": 17,
        "Cache hit": 10,
        "Failed requests": 3,
        "Cached Completion Tokens": 100,
        "Generated Completion Tokens": 500,
      },
    ]);
    expect(summary.cachedCompletionTokens).toBe(100);
  });

  it("returns empty totals for no rows", () => {
    expect(summarizeCacheActivity([])).toEqual({
      chartData: [],
      cacheHits: 0,
      llmApiRequests: 0,
      failedRequests: 0,
      cachedCompletionTokens: 0,
    });
  });
});
