import { describe, expect, it } from "vitest";

import type { components } from "@/lib/http/schema";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  type DailyActivityAggregatedResponse,
  type KeySpendActivityRow,
  toDailyData,
} from "./dailyActivityApi";
import {
  boundedActivityMetrics,
  keyDetailFromResponse,
  mergeKeyActivityPages,
  overallUsageMetrics,
  reportingKeyId,
  storedKeyId,
} from "./keyActivityData";

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
  it("keeps exact totals and an explicit remainder distinct from a real model named Other", () => {
    const response: DailyActivityAggregatedResponse = {
      ...aggregatedResponse,
      model_breakdown: {
        limit: 20,
        total_models: 137,
        models: Array.from({ length: 20 }, (_, index) => ({
          model: index === 0 ? "Other" : `model-${index}`,
          metrics: completeMetrics,
        })),
        other: {
          model_count: 117,
          metrics: { ...completeMetrics, spend: 475, api_requests: 160, total_tokens: 560 },
        },
      },
    };
    const { metrics, otherModels } = boundedActivityMetrics(response);
    expect(metrics.total_spend).toBe(500);
    expect(metrics.total_requests).toBe(200);
    expect(metrics.total_tokens).toBe(700);
    expect(metrics.top_models).toHaveLength(20);
    expect(metrics.top_models?.[0]).toMatchObject({ model: "Other", spend: 1.25, requests: 2 });
    const expectedRemainder = { model_count: 117, spend: 475, requests: 160, tokens: 560 };
    expect(otherModels).toMatchObject(expectedRemainder);
  });

  it("uses exact daily totals without a model summary for non-key activity", () => {
    const { metrics, otherModels } = boundedActivityMetrics(aggregatedResponse);
    expect(metrics.total_spend).toBe(500);
    expect(metrics.daily_data).toHaveLength(1);
    expect(metrics.top_models).toEqual([]);
    expect(otherModels).toBeUndefined();
  });

  it.each([
    ["", "unattributed"],
    ["unattributed", "raw:unattributed"],
    ["raw:key", "raw:raw:key"],
    ["raw:raw:key", "raw:raw:raw:key"],
    ["ordinary-hash", "ordinary-hash"],
    [" ", " "],
  ])("round trips stored key %s through reporting", (key, bucket) => {
    expect(reportingKeyId(key)).toBe(bucket);
    expect(storedKeyId(bucket)).toBe(key);
  });

  it("retains unattributed totals and unknown models in key details", () => {
    const unattributed = {
      metrics: completeMetrics,
      metadata: { key_alias: "Unattributed", key_exists: false },
    };
    const response: DailyActivityAggregatedResponse = {
      ...aggregatedResponse,
      results: [
        {
          date: "2026-09-27",
          metrics: completeMetrics,
          breakdown: {
            api_keys: { unattributed },
            models: {
              "unknown-model": {
                metrics: completeMetrics,
                metadata: {},
                api_key_breakdown: { unattributed },
              },
            },
          },
        },
      ],
    };

    const detail = keyDetailFromResponse(response, "unattributed", []);
    expect(detail?.total_requests).toBe(2);
    expect(detail?.total_spend).toBe(1.25);
    expect(detail?.top_models.map((model) => model.model)).toEqual(["unknown-model"]);
    expect(detail?.daily_data[0].metrics.api_requests).toBe(2);
  });

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

  it("plots descending API days chronologically without reordering the cached response", () => {
    const response = {
      ...aggregatedResponse,
      results: ["2026-09-29", "2026-09-27", "2026-09-28"].map((date, index) => ({
        ...aggregatedResponse.results[0],
        date,
        metrics: { ...completeMetrics, api_requests: index + 10 },
      })),
    };
    const summary = overallUsageMetrics(toDailyData(response), response.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA);
    expect(summary.daily_data.map((day) => [day.date, day.metrics.api_requests])).toEqual([
      ["2026-09-27", 11],
      ["2026-09-28", 12],
      ["2026-09-29", 10],
    ]);
    expect(response.results.map((day) => day.date)).toEqual(["2026-09-29", "2026-09-27", "2026-09-28"]);
    expect(summary.total_requests).toBe(200);
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
