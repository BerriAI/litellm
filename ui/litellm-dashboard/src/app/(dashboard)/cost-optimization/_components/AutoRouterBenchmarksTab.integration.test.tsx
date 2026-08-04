import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

const mockCall = vi.fn();

vi.mock("@/components/networking", () => ({
  autoRouterBenchmarksCall: (...args: unknown[]) => mockCall(...args),
}));

import AutoRouterBenchmarksTab from "./AutoRouterBenchmarksTab";
import type { AutoRouterCacheBenchmark, AutoRouterGroupBenchmark } from "./autoRouterBenchmarks";

const cache = (overrides: Partial<AutoRouterCacheBenchmark> = {}): AutoRouterCacheBenchmark => ({
  ttl_seconds: 3600,
  usage_coverage_pct: 99.6,
  hit_rate_pct: 93.3,
  turns: 818,
  hits: 763,
  same_model_turns: 400,
  same_model_hits: 391,
  first_visit_turns: 37,
  first_visit_hits: 9,
  return_turns: 381,
  return_hits: 311,
  same_model_hit_rate_pct: 97.7,
  first_visit_hit_rate_pct: 24.3,
  return_hit_rate_pct: 81.6,
  stale_miss_share_pct: 27.1,
  warming_savable_miss_pct: 4.0,
  warming_break_even_pct: 5.0,
  stale_return_misses: 19,
  savable_return_misses: 2,
  warming_rescued_spend: 6.76,
  warming_replay_spend: 3.91,
  warming_net_spend: 2.85,
  ...overrides,
});

const group = (overrides: Partial<AutoRouterGroupBenchmark> = {}): AutoRouterGroupBenchmark => ({
  model_group: "claude-auto",
  router_kind: "complexity",
  baseline_model: "anthropic/claude-opus-4-8",
  sessions: 94,
  turns: 3074,
  avg_turns_per_session: 32.7,
  avg_session_length_seconds: 7560,
  total_tokens: 498_200_000,
  avg_tokens_per_session: 5_300_000,
  actual_spend: 359.86,
  baseline_spend: 2534.45,
  savings: 2174.59,
  savings_pct: 85.8,
  cache: cache(),
  ...overrides,
});

const renderTab = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <AutoRouterBenchmarksTab accessToken="sk-test" />
    </QueryClientProvider>,
  );
};

describe("AutoRouterBenchmarksTab", () => {
  it("leads with total estimated savings, before the three session-shape metrics", async () => {
    mockCall.mockResolvedValue({ start_date: "2026-07-02", end_date: "2026-08-01", groups: [group()] });
    renderTab();

    await waitFor(() => expect(screen.getByText("Total estimated savings")).toBeInTheDocument());
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

  it("pairs the savings with the session count it was earned over", async () => {
    mockCall.mockResolvedValue({ start_date: "2026-07-02", end_date: "2026-08-01", groups: [group()] });
    renderTab();

    await waitFor(() => expect(screen.getByText("Sessions on auto-router")).toBeInTheDocument());
    expect(screen.getByText("94")).toBeInTheDocument();
    expect(screen.getByText("3,074 turns")).toBeInTheDocument();
    expect(screen.getByText("Saved per session")).toBeInTheDocument();
    expect(screen.getByText("$23.13")).toBeInTheDocument();
    expect(screen.getByText("Auto-routers in scope")).toBeInTheDocument();
  });

  it("says n/a for saved-per-session when there is no baseline to divide", async () => {
    mockCall.mockResolvedValue({
      start_date: "2026-07-02",
      end_date: "2026-08-01",
      groups: [group({ baseline_model: null, actual_spend: 325.21, baseline_spend: 325.21, savings: 0 })],
    });
    renderTab();

    await waitFor(() => expect(screen.getByText("Saved per session")).toBeInTheDocument());
    expect(screen.getByText("n/a")).toBeInTheDocument();
  });

  it("renders the headline numbers the tiles exist for", async () => {
    mockCall.mockResolvedValue({ start_date: "2026-07-02", end_date: "2026-08-01", groups: [group()] });
    renderTab();

    await waitFor(() => expect(screen.getByText("$2,174.59")).toBeInTheDocument());
    expect(screen.getByText("32.7")).toBeInTheDocument();
    expect(screen.getByText("2.1h")).toBeInTheDocument();
    expect(screen.getByText("5.3M")).toBeInTheDocument();
    expect(screen.getByText("-86%")).toBeInTheDocument();
    expect(screen.getByText(/\$359\.86 routed/)).toBeInTheDocument();
    expect(screen.getByText(/\$2,534\.45 all-claude-opus-4-8 baseline/)).toBeInTheDocument();
  });

  it("shows a cost increase as a positive delta rather than a saving", async () => {
    mockCall.mockResolvedValue({
      start_date: "2026-07-02",
      end_date: "2026-08-01",
      groups: [group({ actual_spend: 120, baseline_spend: 100, savings: -20, savings_pct: -20 })],
    });
    renderTab();

    await waitFor(() => expect(screen.getByText("-$20.00")).toBeInTheDocument());
    expect(screen.getByText("+20%")).toBeInTheDocument();
  });

  it("says the savings were not measured when no router declares a baseline", async () => {
    mockCall.mockResolvedValue({
      start_date: "2026-07-02",
      end_date: "2026-08-01",
      groups: [group({ baseline_model: null, actual_spend: 325.21, baseline_spend: 325.21, savings: 0 })],
    });
    renderTab();

    await waitFor(() => expect(screen.getByText("Not measured")).toBeInTheDocument());
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
  });

  it("renders all three cache buckets with their turn counts and hit rates", async () => {
    mockCall.mockResolvedValue({ start_date: "2026-07-02", end_date: "2026-08-01", groups: [group()] });
    renderTab();

    await waitFor(() => expect(screen.getByText("Same model")).toBeInTheDocument());
    expect(screen.getByText("First visit")).toBeInTheDocument();
    expect(screen.getByText("Return to tier")).toBeInTheDocument();
    expect(screen.getByText("400")).toBeInTheDocument();
    expect(screen.getByText("37")).toBeInTheDocument();
    expect(screen.getByText("381")).toBeInTheDocument();
    expect(screen.getByText("97.8%")).toBeInTheDocument();
    expect(screen.getByText("24.3%")).toBeInTheDocument();
    expect(screen.getByText("81.6%")).toBeInTheDocument();
  });

  it("recomputes each bucket rate from its counts instead of trusting the payload's rate", async () => {
    mockCall.mockResolvedValue({
      start_date: "2026-07-02",
      end_date: "2026-08-01",
      groups: [group({ cache: cache({ return_turns: 200, return_hits: 100, return_hit_rate_pct: 99.9 }) })],
    });
    renderTab();

    await waitFor(() => expect(screen.getByText("Return to tier")).toBeInTheDocument());
    expect(screen.getByText("50.0%")).toBeInTheDocument();
    expect(screen.queryByText("99.9%")).not.toBeInTheDocument();
  });

  it("shows the warming estimate with its two sides and the break-even marker", async () => {
    mockCall.mockResolvedValue({ start_date: "2026-07-02", end_date: "2026-08-01", groups: [group()] });
    renderTab();

    await waitFor(() => expect(screen.getByText("Cache writes rescued")).toBeInTheDocument());
    expect(screen.getByText("$6.76")).toBeInTheDocument();
    expect(screen.getByText("-$3.91")).toBeInTheDocument();
    expect(screen.getByText("$2.85")).toBeInTheDocument();
    expect(screen.getByText(/break-even ≈ 5% at 1h/)).toBeInTheDocument();
  });

  it("hides the caching section when no router reported cache usage", async () => {
    mockCall.mockResolvedValue({
      start_date: "2026-07-02",
      end_date: "2026-08-01",
      groups: [group({ cache: null })],
    });
    renderTab();

    await waitFor(() => expect(screen.getByText("Total estimated savings")).toBeInTheDocument());
    expect(screen.queryByText("Auto-router prompt caching")).not.toBeInTheDocument();
  });

  it("says so when there are no auto-router sessions at all", async () => {
    mockCall.mockResolvedValue({ start_date: "2026-07-02", end_date: "2026-08-01", groups: [] });
    renderTab();

    await waitFor(() =>
      expect(screen.getByText(/No auto-router sessions in the last 30 days yet/)).toBeInTheDocument(),
    );
  });

  it("degrades to a message when the endpoint is unavailable", async () => {
    mockCall.mockRejectedValue(new Error("403"));
    renderTab();

    await waitFor(() =>
      expect(screen.getByText(/Auto-router benchmarks are unavailable right now/)).toBeInTheDocument(),
    );
  });

  it("labels the default selection instead of leaking the __all__ sentinel", async () => {
    mockCall.mockResolvedValue({ start_date: "2026-07-02", end_date: "2026-08-01", groups: [group()] });
    renderTab();

    await waitFor(() => expect(screen.getByText("All auto-routers")).toBeInTheDocument());
    expect(screen.queryByText("__all__")).not.toBeInTheDocument();
  });

  it("requests exactly a thirty day window", async () => {
    mockCall.mockResolvedValue({ start_date: "2026-07-02", end_date: "2026-08-01", groups: [group()] });
    renderTab();

    await waitFor(() => expect(mockCall).toHaveBeenCalled());
    const [, start, end] = mockCall.mock.calls[0] as [string, string, string];
    const days = (Date.parse(end) - Date.parse(start)) / (24 * 60 * 60 * 1000);
    expect(days).toBe(30);
  });
});
