import { describe, expect, it } from "vitest";
import type { DailyData, MetricWithMetadata, SpendMetrics } from "@/components/UsagePage/types";
import { rankCustomerModels, rankCustomerUsage, type CustomerUsageRow } from "./customerUsageData";
const metrics = (spend: number, tokens: number, requests: number): SpendMetrics => ({
  spend,
  prompt_tokens: tokens,
  completion_tokens: 0,
  total_tokens: tokens,
  api_requests: requests,
  successful_requests: requests,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const model = (spend: number, tokens: number, requests: number): MetricWithMetadata => ({
  metrics: metrics(spend, tokens, requests),
  metadata: {},
  api_key_breakdown: {},
});

const day = (
  date: string,
  modelGroups: Record<string, MetricWithMetadata>,
  models: Record<string, MetricWithMetadata> = {},
): DailyData => ({
  date,
  metrics: metrics(0, 0, 0),
  breakdown: {
    models,
    model_groups: modelGroups,
    providers: {},
    mcp_servers: {},
    api_keys: {},
    entities: {},
  },
});

const totals = { spend: 10, tokens: 1_000, requests: 20 };

describe("rankCustomerUsage", () => {
  it("should sort by the selected metric and calculate shares against overall customer totals", () => {
    const rows = [
      { id: "assistant", label: "Assistant", spend: 3, tokens: 100, requests: 8 },
      { id: "search", label: "Search", spend: 1, tokens: 500, requests: 2 },
    ];

    expect(rankCustomerUsage(rows, "spend", totals).map((row) => [row.label, row.share])).toEqual([
      ["Assistant", 0.3],
      ["Search", 0.1],
    ]);
    expect(rankCustomerUsage(rows, "tokens", totals).map((row) => [row.label, row.share])).toEqual([
      ["Search", 0.5],
      ["Assistant", 0.1],
    ]);
    expect(rankCustomerUsage(rows, "requests", totals).map((row) => [row.label, row.share])).toEqual([
      ["Assistant", 0.4],
      ["Search", 0.1],
    ]);
  });

  it("should preserve zero-cost token usage while removing rows with no activity", () => {
    const freeUsage = { id: "assistant", label: "Assistant", spend: 0, tokens: 200, requests: 3 };
    const inactive = { id: "inactive", label: "Inactive", spend: 0, tokens: 0, requests: 0 };

    expect(rankCustomerUsage([inactive, freeUsage], "tokens", totals)).toEqual([{ ...freeUsage, share: 0.2 }]);
    expect(rankCustomerUsage([freeUsage], "spend", totals)).toEqual([{ ...freeUsage, share: 0 }]);
  });

  it("should use stable label and id order for ties without modifying the source rows", () => {
    const rows: readonly Omit<CustomerUsageRow, "share">[] = Object.freeze([
      { id: "b", label: "Search", spend: 1, tokens: 100, requests: 1 },
      { id: "c", label: "Assistant", spend: 1, tokens: 100, requests: 1 },
      { id: "a", label: "Assistant", spend: 1, tokens: 100, requests: 1 },
    ]);
    expect(rankCustomerUsage(rows, "tokens", totals).map((row) => row.id)).toEqual(["a", "c", "b"]);
    expect(rows.map((row) => row.id)).toEqual(["b", "c", "a"]);
  });

  it("should return finite zero shares when the overall metric is zero", () => {
    const rows = [{ id: "assistant", label: "Assistant", spend: 0, tokens: 200, requests: 1 }];
    expect(rankCustomerUsage(rows, "spend", { spend: 0, tokens: 200, requests: 1 })[0].share).toBe(0);
    expect(rankCustomerUsage([], "spend", totals)).toEqual([]);
  });
});

describe("rankCustomerModels", () => {
  it("should sum models across days without adding model and model-group totals together", () => {
    const days = [
      day("2026-11-01", { assistant: model(2, 100, 3) }, { assistant: model(2, 100, 3) }),
      day("2026-11-02", { assistant: model(1, 200, 2), search: model(0, 400, 1) }),
      day("2026-11-03", {}, { assistant: model(1, 100, 1) }),
    ];
    expect(rankCustomerModels(days, "tokens", totals)).toEqual([
      { id: "assistant", label: "assistant", spend: 4, tokens: 400, requests: 6, share: 0.4 },
      { id: "search", label: "search", spend: 0, tokens: 400, requests: 1, share: 0.4 },
    ]);
  });

  it("should return no model rows for an empty date range", () => {
    expect(rankCustomerModels([], "spend", totals)).toEqual([]);
  });
});
