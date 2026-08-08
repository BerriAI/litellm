import { describe, expect, it } from "vitest";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import { mergeDailyResults } from "./mergeDailyActivity";

const metrics = (overrides: Partial<SpendMetrics> = {}): SpendMetrics => ({
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  ...overrides,
});

const day = (date: string, spend: number, teamSpend: Record<string, number>, keySpend: number): DailyData => ({
  date,
  metrics: metrics({ spend, total_tokens: spend * 10, api_requests: 1 }),
  breakdown: {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {
      "sk-a": { metrics: metrics({ spend: keySpend }), metadata: { key_alias: "a", team_id: "team-1" } },
    },
    entities: Object.fromEntries(
      Object.entries(teamSpend).map(([team, value]) => [
        team,
        {
          metrics: metrics({ spend: value, total_tokens: value * 10 }),
          metadata: { team_alias: team },
          api_key_breakdown: {
            "sk-a": { metrics: metrics({ spend: value }), metadata: { key_alias: "a", team_id: team } },
          },
        },
      ]),
    ),
  },
});

describe("mergeDailyResults", () => {
  it("keeps one entry per date when a date straddles a page boundary", () => {
    const pageOne = [day("2026-06-26", 5, { "team-1": 5 }, 5), day("2026-06-25", 22.38, { "team-1": 22.38 }, 22.38)];
    const pageTwo = [day("2026-06-25", 14.52, { "team-1": 14.52 }, 14.52), day("2026-06-24", 3, { "team-1": 3 }, 3)];

    const merged = mergeDailyResults(pageOne, pageTwo);

    expect(merged.map((d) => d.date)).toEqual(["2026-06-26", "2026-06-25", "2026-06-24"]);
    const splitDay = merged.find((d) => d.date === "2026-06-25")!;
    expect(splitDay.metrics.spend).toBeCloseTo(36.9, 10);
    expect(splitDay.metrics.total_tokens).toBeCloseTo(369, 10);
    expect(splitDay.metrics.api_requests).toBe(2);
  });

  it("merges every breakdown bucket of a split date instead of dropping one page's share", () => {
    const merged = mergeDailyResults(
      [day("2026-06-25", 10, { "team-1": 6, "team-2": 4 }, 10)],
      [day("2026-06-25", 5, { "team-2": 5 }, 5)],
    );

    const { entities, api_keys } = merged[0].breakdown;
    expect(entities["team-1"].metrics.spend).toBeCloseTo(6, 10);
    expect(entities["team-2"].metrics.spend).toBeCloseTo(9, 10);
    expect(entities["team-2"].api_key_breakdown["sk-a"].metrics.spend).toBeCloseTo(9, 10);
    expect(api_keys["sk-a"].metrics.spend).toBeCloseTo(15, 10);
  });

  it("preserves the per-day total across pages so day sums match the response metadata", () => {
    const pages = [
      [day("2026-06-25", 22.38, { "team-1": 22.38 }, 22.38)],
      [day("2026-06-25", 14.52, { "team-1": 14.52 }, 14.52)],
      [day("2026-06-24", 3, { "team-1": 3 }, 3)],
    ];

    const merged = pages.reduce<DailyData[]>((acc, page) => mergeDailyResults(acc, page), []);

    expect(merged).toHaveLength(2);
    expect(merged.reduce((total, d) => total + d.metrics.spend, 0)).toBeCloseTo(39.9, 10);
  });

  it("leaves distinct dates untouched", () => {
    const pageOne = [day("2026-06-26", 5, { "team-1": 5 }, 5)];
    const pageTwo = [day("2026-06-25", 7, { "team-1": 7 }, 7)];

    expect(mergeDailyResults(pageOne, pageTwo)).toEqual([...pageOne, ...pageTwo]);
  });
});
