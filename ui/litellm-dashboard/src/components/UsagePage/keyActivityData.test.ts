import { describe, expect, it } from "vitest";

import type { components } from "@/lib/http/schema";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  type DailyActivityAggregatedResponse,
  type KeySpendActivityRow,
  toDailyData,
} from "./dailyActivityApi";
import { keyDetailFromResponse, mergeKeyActivityPages, overallUsageMetrics } from "./keyActivityData";

const completeMetrics: components["schemas"]["SpendMetrics"] = {
  api_requests: 2,
  autorouter_savings_spend: 0,
  cache_creation_input_tokens: 0,
  cache_read_input_tokens: 0,
  completion_tokens: 3,
  compression_saved_tokens: 0,
  compression_savings_spend: 0,
  failed_requests: 0,
  flat_cost: 0,
  gateway_injected_caching_savings_spend: 0,
  prompt_caching_savings_spend: 0,
  prompt_tokens: 4,
  spend: 1.25,
  successful_requests: 2,
  timed_requests: 0,
  total_response_time_ms: 0,
  total_tokens: 7,
};

const apiKeyActivity = {
  metrics: completeMetrics,
  metadata: { key_alias: "visible-key", team_id: null },
};

const aggregatedResponse: DailyActivityAggregatedResponse = {
  metadata: {
    ...EMPTY_DAILY_ACTIVITY_METADATA,
    total_api_requests: 200,
    total_successful_requests: 198,
    total_failed_requests: 2,
    total_tokens: 700,
    total_prompt_tokens: 400,
    total_completion_tokens: 300,
    total_spend: 500,
  },
  results: [
    {
      date: "2026-09-27",
      metrics: completeMetrics,
      breakdown: {
        api_keys: { "key-hash": apiKeyActivity },
        models: {
          "gpt-4o-mini": {
            metrics: completeMetrics,
            metadata: {},
            api_key_breakdown: { "key-hash": apiKeyActivity },
          },
        },
      },
    },
  ],
};

const pageRow = (api_key: string): KeySpendActivityRow => ({
  api_key,
  metrics: {
    api_requests: 1,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    completion_tokens: 2,
    failed_requests: 0,
    prompt_tokens: 3,
    spend: 1,
    successful_requests: 1,
    total_tokens: 5,
  },
  metadata: { key_alias: api_key, team_id: null },
});

describe("key activity data", () => {
  it("builds overall totals from metadata and daily data from complete daily metrics", () => {
    const summary = overallUsageMetrics(
      toDailyData(aggregatedResponse),
      aggregatedResponse.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA,
    );

    expect(summary.total_requests).toBe(200);
    expect(summary.total_spend).toBe(500);
    expect(summary.daily_data).toStrictEqual([
      {
        date: "2026-09-27",
        metrics: {
          prompt_tokens: 4,
          completion_tokens: 3,
          total_tokens: 7,
          api_requests: 2,
          spend: 1.25,
          successful_requests: 2,
          failed_requests: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
          avg_response_time_ms: null,
        },
      },
    ]);
  });

  it("appends pages without duplicate keys and compares the server offset to the total", () => {
    const merged = mergeKeyActivityPages(
      [pageRow("key-a"), pageRow("key-b")],
      [pageRow("key-b"), pageRow("key-c")],
      5,
      2,
    );

    expect(merged.rows.map((row) => row.api_key)).toStrictEqual(["key-a", "key-b", "key-c"]);
    expect(merged.nextOffset).toBe(4);
    expect(merged.hasMore).toBe(true);
    expect(mergeKeyActivityPages(merged.rows, [pageRow("key-d")], 5, 4).hasMore).toBe(false);
  });

  it("advances past a page of already loaded keys and stops only on an empty page", () => {
    const current = [pageRow("key-a"), pageRow("key-b")];
    const duplicates = mergeKeyActivityPages(current, [pageRow("key-a"), pageRow("key-b")], 5, 2);
    expect(duplicates).toEqual({ rows: current, nextOffset: 4, hasMore: true });
    expect(mergeKeyActivityPages(current, [], 5, 2)).toEqual({ rows: current, nextOffset: 2, hasMore: false });
  });

  it("builds full daily details and top models for the requested key", () => {
    const detail = keyDetailFromResponse(aggregatedResponse, "key-hash", []);

    expect(detail?.total_requests).toBe(2);
    expect(detail?.daily_data).toStrictEqual([
      {
        date: "2026-09-27",
        metrics: {
          prompt_tokens: 4,
          completion_tokens: 3,
          total_tokens: 7,
          api_requests: 2,
          spend: 1.25,
          successful_requests: 2,
          failed_requests: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
          avg_response_time_ms: null,
        },
      },
    ]);
    expect(detail?.top_models.map((model) => model.model)).toStrictEqual(["gpt-4o-mini"]);
  });
});
