import { describe, expect, it } from "vitest";

import {
  ALL_ROUTERS,
  bucketRows,
  bucketTurnsTotal,
  durationLabel,
  expiredMissShare,
  groupKey,
  groupLabel,
  pctLabel,
  viewFor,
  type AutoRouterBenchmarkGroup,
  type AutoRouterBenchmarksResponse,
  type AutoRouterCacheStats,
} from "./autoRouterBenchmarks";

const cache = (overrides: Partial<AutoRouterCacheStats> = {}): AutoRouterCacheStats => ({
  coverage_pct: 99.6,
  hit_rate_pct: 93.3,
  same_model: { turns: 400, hits: 391, hit_rate_pct: 97.7 },
  first_visit: { turns: 37, hits: 9, hit_rate_pct: 24.3 },
  return_to_tier: { turns: 381, hits: 311, hit_rate_pct: 81.6 },
  unordered_turns: 0,
  return_misses_expired: 19,
  return_misses_within_ttl: 51,
  return_misses_unknown: 0,
  ttl_5m_turns: 0,
  ttl_1h_turns: 818,
  ...overrides,
});

const totals = (overrides: Partial<AutoRouterBenchmarkGroup> = {}) => ({
  sessions: 94,
  turns: 3073,
  avg_turns_per_session: 32.7,
  avg_session_seconds: 7560,
  avg_tokens_per_session: 5_300_000,
  spend: 359.86,
  saved_spend: 2174.59,
  baseline_spend: 2534.45,
  saved_pct: 85.8,
  saved_per_session: 23.13,
  cache: cache(),
  ...overrides,
});

const group = (overrides: Partial<AutoRouterBenchmarkGroup> = {}): AutoRouterBenchmarkGroup => ({
  router_name: "claude-auto",
  router_type: "complexity",
  ...totals(),
  ...overrides,
});

const response = (groups: AutoRouterBenchmarkGroup[]): AutoRouterBenchmarksResponse => ({
  start_date: "2026-07-06",
  end_date: "2026-08-05",
  routers_in_scope: groups.length,
  totals: totals(),
  groups,
});

describe("viewFor", () => {
  it("maps the all-routers selection to the server totals, never a client sum", () => {
    const data = response([group(), group({ router_name: "gpt-auto", sessions: 7 })]);
    const view = viewFor(data, ALL_ROUTERS);
    expect(view.stats).toBe(data.totals);
    expect(view.label).toBe("All auto-routers");
  });

  it("maps a selected router to that group's slice with a scope of one", () => {
    const other = group({ router_name: "gpt-auto", sessions: 7, saved_spend: 12.5 });
    const data = response([group(), other]);
    const view = viewFor(data, groupKey(other));
    expect(view.stats).toBe(other);
    expect(view.label).toBe("gpt-auto");
  });

  it("falls back to the all-routers view when the selected key no longer exists", () => {
    const data = response([group()]);
    const view = viewFor(data, "vanished complexity");
    expect(view.stats).toBe(data.totals);
    expect(view.label).toBe("All auto-routers");
  });

  it("distinguishes two groups sharing an alias by their router type", () => {
    const a = group({ router_type: "complexity" });
    const b = group({ router_type: "adaptive" });
    const data = response([a, b]);
    expect(groupKey(a)).not.toBe(groupKey(b));
    expect(viewFor(data, groupKey(b)).stats).toBe(b);
    expect(viewFor(data, groupKey(b)).label).toBe("claude-auto (adaptive)");
  });
});

describe("groupLabel", () => {
  it("uses the bare alias when it is unique", () => {
    const groups = [group(), group({ router_name: "gpt-auto" })];
    expect(groupLabel(groups[0], groups)).toBe("claude-auto");
  });

  it("appends the router type only when the alias is duplicated", () => {
    const groups = [group({ router_type: "complexity" }), group({ router_type: "adaptive" })];
    expect(groupLabel(groups[0], groups)).toBe("claude-auto (complexity)");
    expect(groupLabel(groups[1], groups)).toBe("claude-auto (adaptive)");
  });
});

describe("bucketRows", () => {
  it("keeps the three buckets summing to the bucketed turn total", () => {
    const stats = cache();
    const rows = bucketRows(stats);
    expect(rows.map((r) => r.turns)).toEqual([400, 37, 381]);
    expect(bucketTurnsTotal(stats)).toBe(818);
  });

  it("renders the server's per-bucket rates as-is", () => {
    expect(bucketRows(cache()).map((r) => r.hitRatePct)).toEqual([97.7, 24.3, 81.6]);
  });

  it("derives each bucket's share of the measured turns", () => {
    expect(bucketRows(cache()).map((r) => r.sharePct)).toEqual([49, 5, 47]);
  });

  it("reports zero shares instead of dividing by zero when nothing was bucketed", () => {
    const empty = { turns: 0, hits: 0, hit_rate_pct: 0 };
    const rows = bucketRows(cache({ same_model: empty, first_visit: empty, return_to_tier: empty }));
    expect(rows.map((r) => r.sharePct)).toEqual([0, 0, 0]);
  });
});

describe("expiredMissShare", () => {
  it("computes the expired share over every measured turn, not just return-to-tier misses", () => {
    expect(expiredMissShare(cache())).toBeCloseTo((100 * 19) / 818);
  });

  it("is zero, not absent, when every return turn hit", () => {
    expect(
      expiredMissShare(cache({ return_to_tier: { turns: 10, hits: 10, hit_rate_pct: 100 }, return_misses_expired: 0 })),
    ).toBe(0);
  });

  it("is absent only when no turns were measured at all", () => {
    const empty = { turns: 0, hits: 0, hit_rate_pct: 0 };
    const nothingMeasured = {
      same_model: empty,
      first_visit: empty,
      return_to_tier: empty,
      return_misses_expired: 0,
    };
    expect(expiredMissShare(cache(nothingMeasured))).toBeNull();
  });
});

describe("formatting", () => {
  it("renders session length in the largest sensible unit", () => {
    expect(durationLabel(42)).toBe("42s");
    expect(durationLabel(150)).toBe("2.5m");
    expect(durationLabel(7560)).toBe("2.1h");
  });

  it("renders percentages at the requested precision", () => {
    expect(pctLabel(93.3)).toBe("93.3%");
    expect(pctLabel(85.8, 0)).toBe("86%");
  });
});
