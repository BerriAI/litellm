import { describe, expect, test } from "vitest";
import type { DailyData, KeyMetricWithMetadata, MetricWithMetadata, SpendMetrics } from "@/components/UsagePage/types";
import { customerStatus, monthRange, rankBySpend, usageByCustomer } from "./customerPortalData";

const metrics = (spend: number, requests = 1, prompt = 10, completion = 5): SpendMetrics => ({
  spend,
  prompt_tokens: prompt,
  completion_tokens: completion,
  total_tokens: prompt + completion,
  api_requests: requests,
  successful_requests: requests,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const key = (alias: string | null, spend: number, requests = 1): KeyMetricWithMetadata => ({
  metrics: metrics(spend, requests),
  metadata: { key_alias: alias, team_id: null },
});

const entity = (spend: number, keys: Record<string, KeyMetricWithMetadata>, requests = 1): MetricWithMetadata => ({
  metrics: metrics(spend, requests),
  metadata: {},
  api_key_breakdown: keys,
});

const day = (date: string, entities: Record<string, MetricWithMetadata>): DailyData => ({
  date,
  metrics: metrics(0, 0, 0, 0),
  breakdown: { models: {}, model_groups: {}, mcp_servers: {}, providers: {}, api_keys: {}, entities },
});

describe("usageByCustomer", () => {
  test("sums a customer's spend, requests and tokens across days", () => {
    const usage = usageByCustomer([
      day("2026-11-01", { acme: entity(1.5, {}, 2) }),
      day("2026-11-02", { acme: entity(2.25, {}, 3) }),
    ]);
    const expected = { spend: 3.75, requests: 5, inputTokens: 20, outputTokens: 10 };
    expect(usage.get("acme")).toMatchObject(expected);
  });

  test("keeps customers separate", () => {
    const usage = usageByCustomer([day("2026-11-01", { acme: entity(1, {}), globex: entity(4, {}) })]);
    expect(usage.get("acme")?.spend).toBe(1);
    expect(usage.get("globex")?.spend).toBe(4);
  });

  test("merges the same key across days and sorts keys by spend", () => {
    const usage = usageByCustomer([
      day("2026-11-01", { acme: entity(3, { h1: key("prod", 1), h2: key("staging", 2) }) }),
      day("2026-11-02", { acme: entity(5, { h1: key("prod", 5) }) }),
    ]);
    const prod = { hash: "h1", alias: "prod", spend: 6, requests: 2 };
    const staging = { hash: "h2", alias: "staging", spend: 2, requests: 1 };
    expect(usage.get("acme")?.keys).toEqual([expect.objectContaining(prod), expect.objectContaining(staging)]);
  });

  test("keeps the first non-null alias seen for a key", () => {
    const usage = usageByCustomer([
      day("2026-11-01", { acme: entity(1, { h1: key(null, 1) }) }),
      day("2026-11-02", { acme: entity(1, { h1: key("prod", 1) }) }),
    ]);
    expect(usage.get("acme")?.keys[0]?.alias).toBe("prod");
  });
});

describe("customerStatus", () => {
  test("blocked wins over budget", () => {
    expect(customerStatus({ blocked: true, litellm_budget_table: { max_budget: 1 } }, 5)).toBe("blocked");
  });
  test("over budget once spend reaches max_budget", () => {
    expect(customerStatus({ blocked: false, litellm_budget_table: { max_budget: 10 } }, 10)).toBe("over_budget");
    expect(customerStatus({ blocked: false, litellm_budget_table: { max_budget: 10 } }, 9.99)).toBe("active");
  });
  test("no budget is active regardless of spend", () => {
    expect(customerStatus({ blocked: false, litellm_budget_table: null }, 1000)).toBe("active");
  });
});

describe("rankBySpend", () => {
  const customer = (user_id: string) => ({ user_id, blocked: false, spend: 0, litellm_budget_table: null });
  test("orders customers by this month's spend, highest first, zero for customers without traffic", () => {
    const usage = usageByCustomer([day("2026-11-01", { b: entity(2, {}), c: entity(9, { h1: key("k", 9) }) })]);
    const rows = rankBySpend([customer("a"), customer("b"), customer("c")], usage);
    expect(rows.map((r) => [r.c.user_id, r.spend, r.keys])).toEqual([
      ["c", 9, 1],
      ["b", 2, 0],
      ["a", 0, 0],
    ]);
  });
});

describe("monthRange", () => {
  test("covers the whole UTC month, monthsBack months ago", () => {
    const now = new Date(Date.UTC(2026, 10, 9, 12));
    expect(monthRange(0, now)).toEqual({
      start: new Date(Date.UTC(2026, 10, 1)),
      end: new Date(Date.UTC(2026, 10, 30, 23, 59, 59)),
    });
    expect(monthRange(1, now)).toEqual({
      start: new Date(Date.UTC(2026, 9, 1)),
      end: new Date(Date.UTC(2026, 9, 31, 23, 59, 59)),
    });
  });
  test("crosses a year boundary", () => {
    expect(monthRange(2, new Date(Date.UTC(2026, 0, 15))).start).toEqual(new Date(Date.UTC(2025, 10, 1)));
  });
});
