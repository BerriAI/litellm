import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { EMPTY_DAILY_ACTIVITY_RESPONSE } from "@/components/UsagePage/dailyActivityApi";
import { renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";
import type { AutoRouterBenchmarksResponse } from "./autoRouterBenchmarks";
import CostOptimizationView from "./CostOptimizationView";

const emptyBucket = { turns: 0, hits: 0, hit_rate_pct: 0 };
const emptyBenchmarks: AutoRouterBenchmarksResponse = {
  start_date: "2026-09-14",
  end_date: "2026-10-14",
  routers_in_scope: 0,
  groups: [],
  totals: {
    sessions: 0,
    turns: 0,
    avg_turns_per_session: 0,
    avg_session_seconds: 0,
    avg_tokens_per_session: 0,
    spend: 0,
    savings_estimated_turns: 0,
    savings_estimated_actual_spend: 0,
    savings_estimated_classifier_cost: 0,
    classifier_cost: 0,
    saved_spend: 0,
    baseline_spend: 0,
    saved_pct: 0,
    cache: {
      coverage_pct: 0,
      hit_rate_pct: 0,
      same_model: emptyBucket,
      first_visit: emptyBucket,
      return_to_tier: emptyBucket,
      unordered_turns: 0,
      return_misses_expired: 0,
      return_misses_within_ttl: 0,
      return_misses_unknown: 0,
      ttl_5m_turns: 0,
      ttl_1h_turns: 0,
    },
  },
};

const requestUrl = (input: Parameters<typeof fetch>[0]): URL =>
  new URL(input instanceof Request ? input.url : String(input), "http://localhost:4000");

describe("CostOptimizationView shared date range", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-10-14T12:00:00"));
    vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) => {
      callback({ didTimeout: false, timeRemaining: () => 50 });
      return 0;
    });
    const payload = btoa(JSON.stringify({ key: "sk-test", user_id: "u1", user_role: "proxy_admin" }));
    document.cookie = `token=e30.${payload}.test; path=/`;
    testQueryClient.clear();
  });

  afterEach(() => {
    document.cookie = "token=; max-age=0; path=/";
    testQueryClient.clear();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("shares the initial seven days and a custom wider range across the three date pickers", async () => {
    const mockFetch = vi.fn<typeof fetch>().mockImplementation(async (input) => {
      const path = requestUrl(input).pathname;
      const responses: Readonly<Record<string, unknown>> = {
        "/user/daily/activity/aggregated": EMPTY_DAILY_ACTIVITY_RESPONSE,
        "/litellm/.well-known/litellm-ui-config": { admin_ui_disabled: false },
        "/v1/tool/spend": { by_tool: [], daily: [] },
        "/config/list": [],
        "/v2/model/info": { data: [] },
        "/auto_router/benchmarks": emptyBenchmarks,
        "/user/daily/activity/aggregated/cache_leakage_keys": { api_keys: [] },
        "/cost_optimization/prompt_caching/requests": { requests: [], next_cursor: null },
      };
      const data = responses[path] ?? [];
      return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } });
    });
    vi.stubGlobal("fetch", mockFetch);
    renderWithProviders(<CostOptimizationView accessToken="sk-test" userId="u1" userRole="proxy_admin" />);

    const activityRequests = () =>
      mockFetch.mock.calls
        .map(([input]) => requestUrl(input))
        .filter((url) => url.pathname === "/user/daily/activity/aggregated");
    await waitFor(() => expect(activityRequests()).toHaveLength(1));
    expect(activityRequests()[0].searchParams.get("start_date")).toBe("2026-10-08");
    expect(activityRequests()[0].searchParams.get("end_date")).toBe("2026-10-14");

    const initialRange = /8 Oct, 00:00 - 14 Oct, 23:59/;
    expect(screen.getByRole("button", { name: initialRange })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Prompt Caching" }));
    fireEvent.click(screen.getByRole("button", { name: initialRange }));
    fireEvent.change(screen.getByDisplayValue("2026-10-08"), { target: { value: "2026-09-14" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));

    await waitFor(() => expect(activityRequests().at(-1)?.searchParams.get("start_date")).toBe("2026-09-14"));
    expect(activityRequests().at(-1)?.searchParams.get("end_date")).toBe("2026-10-14");
    const selectedRange = /14 Sep, 00:00 - 14 Oct, 23:59/;
    expect(screen.getByRole("button", { name: selectedRange })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Auto-Router" }));
    expect(screen.getByRole("button", { name: selectedRange })).toBeInTheDocument();
    await waitFor(() =>
      expect(mockFetch).toHaveBeenCalledWith(
        expect.objectContaining({ url: expect.stringContaining("/auto_router/benchmarks?") }),
      ),
    );
    const benchmarkUrl = mockFetch.mock.calls
      .map(([input]) => requestUrl(input))
      .find((url) => url.pathname === "/auto_router/benchmarks");
    expect(benchmarkUrl?.searchParams.get("start_date")).toBe("2026-09-14");
    expect(benchmarkUrl?.searchParams.get("end_date")).toBe("2026-10-14");
    fireEvent.click(screen.getByRole("tab", { name: "Overall" }));
    expect(screen.getByRole("button", { name: selectedRange })).toBeInTheDocument();
  });
});
