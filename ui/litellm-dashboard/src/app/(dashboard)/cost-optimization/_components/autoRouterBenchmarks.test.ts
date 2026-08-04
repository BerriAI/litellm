import { describe, expect, it } from "vitest";

import { toView, usd, type AutoRouterCacheBenchmark, type AutoRouterGroupBenchmark } from "./autoRouterBenchmarks";

const cache = (overrides: Partial<AutoRouterCacheBenchmark> = {}): AutoRouterCacheBenchmark => ({
  ttl_seconds: 3600,
  usage_coverage_pct: 100,
  hit_rate_pct: 90,
  turns: 100,
  hits: 90,
  same_model_turns: 70,
  same_model_hits: 68,
  first_visit_turns: 10,
  first_visit_hits: 2,
  return_turns: 20,
  return_hits: 20,
  same_model_hit_rate_pct: 97.1,
  first_visit_hit_rate_pct: 20,
  return_hit_rate_pct: 100,
  stale_miss_share_pct: 0,
  warming_savable_miss_pct: 4,
  warming_break_even_pct: 5,
  stale_return_misses: 0,
  savable_return_misses: 0,
  warming_rescued_spend: 6.76,
  warming_replay_spend: 3.91,
  warming_net_spend: 2.85,
  ...overrides,
});

const group = (overrides: Partial<AutoRouterGroupBenchmark> = {}): AutoRouterGroupBenchmark => ({
  model_group: "claude-auto",
  router_kind: "complexity",
  baseline_model: "anthropic/claude-opus-4-8",
  sessions: 10,
  turns: 100,
  avg_turns_per_session: 10,
  avg_session_length_seconds: 3600,
  total_tokens: 1_000_000,
  avg_tokens_per_session: 100_000,
  actual_spend: 10,
  baseline_spend: 100,
  savings: 90,
  savings_pct: 90,
  cache: cache(),
  ...overrides,
});

describe("toView", () => {
  it("returns null when there are no groups", () => {
    expect(toView([])).toBeNull();
  });

  it("returns null when the groups hold no sessions, rather than dividing by zero", () => {
    expect(toView([group({ sessions: 0, turns: 0 })])).toBeNull();
  });

  it("divides total turns by total sessions rather than averaging per-router averages", () => {
    const view = toView([
      group({ model_group: "a", sessions: 1, turns: 100 }),
      group({ model_group: "b", sessions: 99, turns: 99 }),
    ]);
    expect(view?.avg_turns_per_session).toBeCloseTo(199 / 100);
  });

  it("weights session length by session count", () => {
    const view = toView([
      group({ model_group: "a", sessions: 1, avg_session_length_seconds: 100 }),
      group({ model_group: "b", sessions: 9, avg_session_length_seconds: 1000 }),
    ]);
    expect(view?.avg_session_length_seconds).toBeCloseTo((100 * 1 + 1000 * 9) / 10);
  });

  it("recomputes savings from summed spend so it stays consistent with the tiles", () => {
    const view = toView([
      group({ model_group: "a", actual_spend: 10, baseline_spend: 100 }),
      group({ model_group: "b", actual_spend: 5, baseline_spend: 20 }),
    ]);
    expect(view?.actual_spend).toBe(15);
    expect(view?.baseline_spend).toBe(120);
    expect(view?.savings).toBe(105);
    expect(view?.savings_pct).toBeCloseTo((100 * 105) / 120);
  });

  it("keeps the sign when routing cost more than the baseline", () => {
    const view = toView([group({ actual_spend: 120, baseline_spend: 100 })]);
    expect(view?.savings).toBe(-20);
    expect(view?.savings_pct).toBeCloseTo(-20);
  });

  it("reports no percentage instead of dividing by a zero baseline", () => {
    expect(toView([group({ actual_spend: 5, baseline_spend: 0 })])?.savings_pct).toBe(0);
  });
});

describe("toView cache combination", () => {
  it("weights the headline hit rate by turns rather than averaging the routers", () => {
    const view = toView([
      group({ model_group: "a", cache: cache({ turns: 1000, hits: 900 }) }),
      group({ model_group: "b", cache: cache({ turns: 10, hits: 0 }) }),
    ]);
    expect(view?.cache?.hit_rate_pct).toBeCloseTo((100 * 900) / 1010);
  });

  it("keeps the three buckets summing to the combined turn count", () => {
    const view = toView([
      group({ model_group: "a" }),
      group({
        model_group: "b",
        cache: cache({ same_model_turns: 5, first_visit_turns: 3, return_turns: 2, turns: 10 }),
      }),
    ]);
    const c = view?.cache;
    expect(c).toBeTruthy();
    expect((c?.same_model_turns ?? 0) + (c?.first_visit_turns ?? 0) + (c?.return_turns ?? 0)).toBe(c?.turns);
  });

  it("recomputes the savable share against every combined miss", () => {
    const view = toView([
      group({ model_group: "a", cache: cache({ turns: 100, hits: 90, savable_return_misses: 5 }) }),
      group({ model_group: "b", cache: cache({ turns: 100, hits: 80, savable_return_misses: 5 }) }),
    ]);
    expect(view?.cache?.warming_savable_miss_pct).toBeCloseTo((100 * 10) / 30);
  });

  it("nets the warming estimate from the summed sides", () => {
    const view = toView([
      group({ model_group: "a", cache: cache({ warming_rescued_spend: 6, warming_replay_spend: 4 }) }),
      group({ model_group: "b", cache: cache({ warming_rescued_spend: 1, warming_replay_spend: 5 }) }),
    ]);
    expect(view?.cache?.warming_net_spend).toBeCloseTo(-2);
  });

  it("flags mixed TTLs so the card does not claim one regime for all of them", () => {
    const view = toView([
      group({ model_group: "a", cache: cache({ ttl_seconds: 300 }) }),
      group({ model_group: "b", cache: cache({ ttl_seconds: 3600 }) }),
    ]);
    expect(view?.mixedTtl).toBe(true);
  });

  it("omits the cache entirely when no router reported one", () => {
    expect(toView([group({ cache: null })])?.cache).toBeNull();
  });
});

describe("toView baseline label", () => {
  it("names the shared baseline when every router used the same one", () => {
    const view = toView([group({ model_group: "a" }), group({ model_group: "b" })]);
    expect(view?.baselineLabel).toBe("anthropic/claude-opus-4-8");
  });

  it("refuses to name one baseline when the routers disagree", () => {
    const view = toView([
      group({ model_group: "a", baseline_model: "anthropic/claude-opus-4-8" }),
      group({ model_group: "b", baseline_model: "openai/gpt-5" }),
    ]);
    expect(view?.baselineLabel).toBe("each router's own baseline");
  });

  it("reports no baseline when the routers declare none, so savings is not shown as zero", () => {
    expect(toView([group({ baseline_model: null })])?.baselineLabel).toBeNull();
  });
});

describe("usd", () => {
  it("sizes and signs off the magnitude so a small loss does not render as -$0.00", () => {
    expect(usd(-0.004)).toBe("-$0.00");
    expect(usd(-12.5)).toBe("-$12.50");
    expect(usd(2174.59)).toBe("$2,174.59");
  });
});

describe("toView saved per session", () => {
  it("divides total savings by total sessions", () => {
    const view = toView([
      group({ model_group: "a", sessions: 60, actual_spend: 10, baseline_spend: 100 }),
      group({ model_group: "b", sessions: 40, actual_spend: 5, baseline_spend: 20 }),
    ]);
    expect(view?.saved_per_session).toBeCloseTo(105 / 100);
  });

  it("goes negative when routing cost more than the baseline", () => {
    expect(toView([group({ sessions: 10, actual_spend: 120, baseline_spend: 100 })])?.saved_per_session).toBeCloseTo(
      -2,
    );
  });
});
