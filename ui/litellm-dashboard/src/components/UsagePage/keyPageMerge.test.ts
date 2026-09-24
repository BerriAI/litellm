import { describe, expect, it } from "vitest";

import { mergeKeyPage } from "./keyPageMerge";
import type { DailyData, KeyMetricWithMetadata, MetricWithMetadata, SpendMetrics } from "./types";

const metrics = (spend: number): SpendMetrics => ({
  spend,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const keyEntry = (spend: number, alias: string): KeyMetricWithMetadata => ({
  metrics: metrics(spend),
  metadata: { key_alias: alias, team_id: null, user_id: null, user_email: null },
});

const modelEntry = (apiKeys: Record<string, KeyMetricWithMetadata>): MetricWithMetadata => ({
  metrics: metrics(0),
  metadata: {},
  api_key_breakdown: apiKeys,
});

const day = (date: string, apiKeys: Record<string, KeyMetricWithMetadata>, model = "gpt-4"): DailyData => ({
  date,
  metrics: metrics(1),
  breakdown: {
    models: { [model]: modelEntry(apiKeys) },
    model_groups: { "group-a": modelEntry(apiKeys) },
    providers: { openai: modelEntry(apiKeys) },
    mcp_servers: {},
    api_keys: apiKeys,
    entities: {},
    endpoints: { "/v1/chat": modelEntry(apiKeys) },
  },
});

const page = (results: DailyData[], next_cursor: string | null) => ({
  results,
  metadata: { total_spend: 42, total_api_keys: 3, next_cursor },
});

describe("mergeKeyPage", () => {
  it("adds keys from the next page to the same day's breakdown", () => {
    const prev = page([day("2026-01-01", { "hash-a": keyEntry(1, "a") })], "c1");
    const next = page([day("2026-01-01", { "hash-b": keyEntry(0.5, "b") })], null);

    const merged = mergeKeyPage(prev, next);

    expect(Object.keys(merged.results[0].breakdown.api_keys)).toEqual(["hash-a", "hash-b"]);
    expect(merged.results[0].breakdown.models["gpt-4"].api_key_breakdown).toHaveProperty("hash-b");
    expect(merged.results[0].breakdown.providers.openai.api_key_breakdown).toHaveProperty("hash-b");
    expect(merged.results[0].breakdown.endpoints?.["/v1/chat"].api_key_breakdown).toHaveProperty("hash-b");
  });

  it("lets the next page win when a key appears on both pages", () => {
    const prev = page([day("2026-01-01", { "hash-a": keyEntry(1, "a") })], "c1");
    const next = page([day("2026-01-01", { "hash-a": keyEntry(9, "a-new") })], null);

    const merged = mergeKeyPage(prev, next);

    expect(merged.results[0].breakdown.api_keys["hash-a"].metadata.key_alias).toBe("a-new");
    expect(merged.results[0].breakdown.models["gpt-4"].api_key_breakdown["hash-a"].metrics.spend).toBe(9);
  });

  it("keeps prev metadata totals and takes next_cursor from next", () => {
    const prev = page([day("2026-01-01", { "hash-a": keyEntry(1, "a") })], "c1");
    const next = page([day("2026-01-01", { "hash-b": keyEntry(0.5, "b") })], null);

    const merged = mergeKeyPage(prev, next);

    expect(merged.metadata.total_spend).toBe(42);
    expect(merged.metadata.total_api_keys).toBe(3);
    expect(merged.metadata.next_cursor).toBeNull();
  });

  it("appends a day that only exists in the next page", () => {
    const prev = page([day("2026-01-01", { "hash-a": keyEntry(1, "a") })], "c1");
    const next = page([day("2026-01-02", { "hash-b": keyEntry(0.5, "b") })], "c2");

    const merged = mergeKeyPage(prev, next);

    expect(merged.results.map((d) => d.date)).toEqual(["2026-01-01", "2026-01-02"]);
    expect(merged.metadata.next_cursor).toBe("c2");
  });

  it("does not mutate prev", () => {
    const prev = page([day("2026-01-01", { "hash-a": keyEntry(1, "a") })], "c1");
    const next = page([day("2026-01-01", { "hash-b": keyEntry(0.5, "b") })], null);

    mergeKeyPage(prev, next);

    expect(Object.keys(prev.results[0].breakdown.api_keys)).toEqual(["hash-a"]);
    expect(prev.results[0].breakdown.models["gpt-4"].api_key_breakdown).not.toHaveProperty("hash-b");
  });
});
