import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/http/client";

vi.mock("./useAutoRouterBenchmarks", () => ({ useAutoRouterBenchmarks: vi.fn() }));
vi.mock("./useAutoRouterQualitySignals", () => ({ useAutoRouterQualitySignals: vi.fn() }));

import AutoRouterBenchmarksTab from "./AutoRouterBenchmarksTab";
import type {
  AutoRouterBenchmarkGroup,
  AutoRouterBenchmarksResponse,
  AutoRouterCacheStats,
} from "./autoRouterBenchmarks";
import type { AutoRouterQualitySignalsResponse } from "./autoRouterQualitySignals";
import { useAutoRouterBenchmarks } from "./useAutoRouterBenchmarks";
import { useAutoRouterQualitySignals } from "./useAutoRouterQualitySignals";

type HookResult = ReturnType<typeof useAutoRouterBenchmarks>;
type QualityHookResult = ReturnType<typeof useAutoRouterQualitySignals>;

const mockHook = (result: { data?: AutoRouterBenchmarksResponse; isPending?: boolean; error?: Error }) => {
  vi.mocked(useAutoRouterBenchmarks).mockReturnValue({
    data: result.data,
    isPending: result.isPending ?? false,
    error: result.error ?? null,
  } as unknown as HookResult);
};

const mockQualityHook = (data?: AutoRouterQualitySignalsResponse) => {
  vi.mocked(useAutoRouterQualitySignals).mockReturnValue({
    data,
    isPending: false,
    error: null,
  } as unknown as QualityHookResult);
};

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

type Totals = AutoRouterBenchmarksResponse["totals"];

const totals = (overrides: Partial<Totals> = {}): Totals => ({
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

const response = (groups: AutoRouterBenchmarkGroup[], shared: Totals = totals()): AutoRouterBenchmarksResponse => ({
  start_date: "2026-07-06",
  end_date: "2026-08-05",
  routers_in_scope: groups.length,
  totals: shared,
  groups,
});

const renderTab = () => render(<AutoRouterBenchmarksTab accessToken="sk-test" />);

const cohort = (
  overrides: Partial<AutoRouterQualitySignalsResponse["totals"]["routed"]> = {},
): AutoRouterQualitySignalsResponse["totals"]["routed"] => ({
  sessions: 120,
  escalation_rate_pct: 6.9,
  abandonment_rate_pct: 2.4,
  ...overrides,
});

const qualityResponse = (
  overrides: Partial<AutoRouterQualitySignalsResponse["totals"]> = {},
): AutoRouterQualitySignalsResponse => ({
  start_date: "2026-07-06",
  end_date: "2026-08-05",
  totals: {
    router_name: null,
    routed: cohort(),
    baseline: cohort({ sessions: 80, escalation_rate_pct: 3.1, abandonment_rate_pct: 2.1 }),
    baseline_unavailable_reason: null,
    ...overrides,
  },
  groups: [],
});

describe("AutoRouterBenchmarksTab", () => {
  beforeEach(() => {
    mockQualityHook(undefined);
  });

  it("leads with total estimated savings, before the three session-shape metrics", () => {
    mockHook({ data: response([group(), group({ router_name: "gpt-auto" })]) });
    renderTab();

    const labels = screen
      .getAllByText(/Total estimated savings|Avg turns per session|Avg session length|Avg tokens per session/)
      .map((node) => node.textContent);
    expect(labels).toEqual([
      "Total estimated savings",
      "Avg turns per session",
      "Avg session length",
      "Avg tokens per session",
    ]);
  });

  it("renders the headline numbers the tiles exist for", () => {
    mockHook({ data: response([group(), group({ router_name: "gpt-auto" })]) });
    renderTab();

    expect(screen.getByText("$2,174.59")).toBeInTheDocument();
    expect(screen.getByText("-86%")).toBeInTheDocument();
    expect(screen.getByText("Actual auto-router spend")).toBeInTheDocument();
    expect(screen.getByText("$359.86")).toBeInTheDocument();
    expect(screen.getByText("Estimated spend at highest-cost model")).toBeInTheDocument();
    expect(screen.getByText("$2,534.45")).toBeInTheDocument();
    expect(screen.getByText("32.7")).toBeInTheDocument();
    expect(screen.getByText("2.1h")).toBeInTheDocument();
    expect(screen.getByText("5.3M")).toBeInTheDocument();
  });

  it("pairs the savings with the session count it was earned over", () => {
    mockHook({ data: response([group(), group({ router_name: "gpt-auto" })]) });
    renderTab();

    expect(screen.getByText("Total sessions")).toBeInTheDocument();
    expect(screen.getByText("94")).toBeInTheDocument();
    expect(screen.getByText("Total turns")).toBeInTheDocument();
    expect(screen.getByText("3,073")).toBeInTheDocument();
    expect(screen.getByText("Avg saved per session")).toBeInTheDocument();
    expect(screen.getByText("$23.13")).toBeInTheDocument();
  });

  it("shows a cost increase as a positive delta rather than a saving", () => {
    const overBaseline = { spend: 120, baseline_spend: 100, saved_spend: -20, saved_pct: -20 };
    const dearer = totals(overBaseline);
    mockHook({ data: response([group(dearer)], dearer) });
    renderTab();

    expect(screen.getByText("+20%")).toBeInTheDocument();
  });

  it("renders all three cache buckets with their turn counts and hit rates", () => {
    mockHook({ data: response([group()]) });
    renderTab();

    expect(screen.getByText("Same model")).toBeInTheDocument();
    expect(screen.getByText("previous turn → same tier")).toBeInTheDocument();
    expect(screen.getByText("First visit")).toBeInTheDocument();
    expect(screen.getByText("previous turn → a tier not used yet")).toBeInTheDocument();
    expect(screen.getByText("Return to tier")).toBeInTheDocument();
    expect(screen.getByText("previous turn → a tier used earlier")).toBeInTheDocument();
    expect(screen.getByText("400")).toBeInTheDocument();
    expect(screen.getByText("37")).toBeInTheDocument();
    expect(screen.getByText("381")).toBeInTheDocument();
    expect(screen.getByText("49%")).toBeInTheDocument();
    expect(screen.getByText("5%")).toBeInTheDocument();
    expect(screen.getByText("47%")).toBeInTheDocument();
    expect(screen.getByText("97.7%")).toBeInTheDocument();
    expect(screen.getByText("24.3%")).toBeInTheDocument();
    expect(screen.getByText("81.6%")).toBeInTheDocument();
  });

  it("summarizes the cache column from the bucketed turns, not the session turns", () => {
    mockHook({ data: response([group()]) });
    renderTab();

    expect(screen.getByText("93.3%")).toBeInTheDocument();
    expect(screen.getByText("818")).toBeInTheDocument();
    expect(screen.getByText(/turns measured/)).toBeInTheDocument();
  });

  it("computes the expired-miss share over every measured turn, not just return-to-tier misses", () => {
    mockHook({ data: response([group()]) });
    renderTab();

    expect(screen.getByText("Expired-miss")).toBeInTheDocument();
    expect(screen.getByText("2.3%")).toBeInTheDocument();
  });

  it("exposes the whole expired-miss row as a focusable tooltip trigger", () => {
    mockHook({ data: response([group()]) });
    renderTab();

    const trigger = screen.getByRole("button", { name: /Expired-miss/ });
    expect(trigger).toHaveTextContent("2.3%");
  });

  it("shows a zero expired-miss share, rather than hiding the row, when every return turn hit", () => {
    const allHits = totals({
      cache: cache({ return_to_tier: { turns: 381, hits: 381, hit_rate_pct: 100 }, return_misses_expired: 0 }),
    });
    mockHook({ data: response([group(allHits)], allHits) });
    renderTab();

    const trigger = screen.getByRole("button", { name: /Expired-miss/ });
    expect(trigger).toHaveTextContent("0.0%");
  });

  it("hides the expired-miss row only when no turns were measured at all", () => {
    const empty = { turns: 0, hits: 0, hit_rate_pct: 0 };
    const nothingMeasured = {
      same_model: empty,
      first_visit: empty,
      return_to_tier: empty,
      return_misses_expired: 0,
    };
    const noTurns = totals({ cache: cache(nothingMeasured) });
    mockHook({ data: response([group(noTurns)], noTurns) });
    renderTab();

    expect(screen.queryByText("Expired-miss")).not.toBeInTheDocument();
  });

  it("mentions out-of-order turns only when there are any", () => {
    const unordered = totals({ cache: cache({ unordered_turns: 12 }) });
    mockHook({ data: response([group(unordered)], unordered) });
    renderTab();

    expect(screen.getByText(/12 turns arrived out of order across pods and are not bucketed/)).toBeInTheDocument();
  });

  it("labels the default selection instead of leaking the __all__ sentinel", () => {
    mockHook({ data: response([group()]) });
    renderTab();

    expect(screen.getByText("All auto-routers")).toBeInTheDocument();
    expect(screen.queryByText("__all__")).not.toBeInTheDocument();
  });

  it("says so while the benchmarks are loading", () => {
    mockHook({ isPending: true });
    renderTab();

    expect(screen.getByText("Loading auto-router usage...")).toBeInTheDocument();
  });

  it("names the admin requirement when the proxy answers 403", () => {
    mockHook({ error: new ApiError("forbidden", 403, {}) });
    renderTab();

    expect(screen.getByText("Auto-router usage is visible to proxy admin roles only")).toBeInTheDocument();
  });

  it("degrades to a message when the endpoint is unavailable", () => {
    mockHook({ error: new ApiError("boom", 500, {}) });
    renderTab();

    expect(screen.getByText("Auto-router usage is unavailable right now")).toBeInTheDocument();
  });

  it("says so when there are no auto-router sessions at all", () => {
    mockHook({ data: response([]) });
    renderTab();

    expect(screen.getByText("No auto-router sessions in this window yet")).toBeInTheDocument();
  });

  it("requests the default thirty day window and widens or narrows it from the picker", () => {
    mockHook({ data: response([group()]) });
    renderTab();

    expect(vi.mocked(useAutoRouterBenchmarks)).toHaveBeenCalledWith("sk-test", "30d");
    expect(screen.getByText("Last 30 days")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "7d" }));
    expect(vi.mocked(useAutoRouterBenchmarks)).toHaveBeenCalledWith("sk-test", "7d");
    expect(screen.getByText("Last 7 days")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "24h" }));
    expect(vi.mocked(useAutoRouterBenchmarks)).toHaveBeenCalledWith("sk-test", "24h");
    expect(screen.getByText("Last 24 hours")).toBeInTheDocument();
  });

  it("keeps the window picker reachable while a window has no sessions", () => {
    mockHook({ data: response([]) });
    renderTab();

    expect(screen.getByRole("tab", { name: "30d" })).toBeInTheDocument();
    expect(screen.getByText("All auto-routers")).toBeInTheDocument();
  });

  describe("quality signals", () => {
    it("renders escalation and abandonment against the non-router baseline", () => {
      mockHook({ data: response([group()]) });
      mockQualityHook(qualityResponse());
      renderTab();

      expect(screen.getByText("Escalation rate")).toBeInTheDocument();
      expect(screen.getByText("6.9%")).toBeInTheDocument();
      expect(screen.getByText("vs. 3.1% on your non-router traffic")).toBeInTheDocument();

      expect(screen.getByText("Stream abandonment")).toBeInTheDocument();
      expect(screen.getByText("2.4%")).toBeInTheDocument();
      expect(screen.getByText("vs. 2.1% on your non-router traffic")).toBeInTheDocument();
    });

    it("does not render the quality card when there is no quality data yet", () => {
      mockHook({ data: response([group()]) });
      mockQualityHook(undefined);
      renderTab();

      expect(screen.queryByText("Escalation rate")).not.toBeInTheDocument();
    });

    it("explains why the baseline is missing instead of showing a misleading rate", () => {
      mockHook({ data: response([group()]) });
      mockQualityHook(
        qualityResponse({
          baseline: null,
          baseline_unavailable_reason: "no_session_ids",
        }),
      );
      renderTab();

      expect(
        screen.getAllByText(
          "Non-router traffic isn't sending session IDs, so it can't be grouped into sessions to compare",
        ),
      ).toHaveLength(2);
    });

    it("explains an insufficient-sessions baseline distinctly from a missing-session-id one", () => {
      mockHook({ data: response([group()]) });
      mockQualityHook(
        qualityResponse({
          baseline: null,
          baseline_unavailable_reason: "insufficient_sessions",
        }),
      );
      renderTab();

      expect(screen.getAllByText("Not enough comparable non-router traffic in this window to compare")).toHaveLength(2);
    });

    it("flags escalation as worse than baseline visually distinctly from a healthy rate", () => {
      mockHook({ data: response([group()]) });
      mockQualityHook(
        qualityResponse({
          routed: cohort({ escalation_rate_pct: 9.0 }),
          baseline: cohort({ escalation_rate_pct: 3.0 }),
        }),
      );
      renderTab();

      expect(screen.getByText("9.0%")).toHaveClass("text-destructive");
    });
  });
});
