import { describe, expect, it } from "vitest";

import type { DailyData, MetricWithMetadata, SpendMetrics } from "@/components/UsagePage/types";

import { singleTeamId, teamTagSpendRows } from "./teamTagSpend";

const metrics = (spend: number, api_requests: number, failed_requests = 0): SpendMetrics => ({
  spend,
  prompt_tokens: 10 * api_requests,
  completion_tokens: 5 * api_requests,
  total_tokens: 15 * api_requests,
  api_requests,
  successful_requests: api_requests - failed_requests,
  failed_requests,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const entity = (spend: number, api_requests: number, failed_requests = 0): MetricWithMetadata => ({
  metrics: metrics(spend, api_requests, failed_requests),
  metadata: {},
  api_key_breakdown: {},
});

const day = (date: string, entities: Record<string, MetricWithMetadata>): DailyData => ({
  date,
  metrics: metrics(0, 0),
  breakdown: { models: {}, model_groups: {}, mcp_servers: {}, providers: {}, api_keys: {}, entities },
});

describe("teamTagSpendRows", () => {
  it("sums each tag across days and orders by spend, highest first", () => {
    const rows = teamTagSpendRows([
      day("2026-10-18", { "cost-center:cc-42": entity(0.5, 2), "project:rag-v2": entity(0.25, 1) }),
      day("2026-10-19", { "cost-center:cc-42": entity(0.25, 1, 1), "env:prod": entity(1, 4) }),
    ]);

    expect(rows).toEqual([
      { tag: "env:prod", spend: 1, api_requests: 4, successful_requests: 4, failed_requests: 0, total_tokens: 60 },
      {
        tag: "cost-center:cc-42",
        spend: 0.75,
        api_requests: 3,
        successful_requests: 2,
        failed_requests: 1,
        total_tokens: 45,
      },
      {
        tag: "project:rag-v2",
        spend: 0.25,
        api_requests: 1,
        successful_requests: 1,
        failed_requests: 0,
        total_tokens: 15,
      },
    ]);
  });

  it("drops tags with no spend and breaks spend ties by tag name", () => {
    const rows = teamTagSpendRows([
      day("2026-10-19", { "b-tag": entity(0.5, 1), "a-tag": entity(0.5, 1), "free-tag": entity(0, 3) }),
    ]);

    expect(rows.map((row) => row.tag)).toEqual(["a-tag", "b-tag"]);
  });

  it("returns no rows when the team sent no tagged spend", () => {
    expect(teamTagSpendRows([day("2026-10-19", {})])).toEqual([]);
    expect(teamTagSpendRows([])).toEqual([]);
  });
});

describe("singleTeamId", () => {
  it("is the team only when exactly one is selected", () => {
    expect(singleTeamId(["team-a"])).toBe("team-a");
    expect(singleTeamId([])).toBeNull();
    expect(singleTeamId(["team-a", "team-b"])).toBeNull();
  });
});
