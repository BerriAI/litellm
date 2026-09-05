import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { ApiError } from "@/lib/http/client";

vi.mock("./useAutoRouterBenchmarks", () => ({ useAutoRouterBenchmarks: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({ useAutoRouters: vi.fn() }));
vi.mock("./ShadowEvalSection", () => ({ default: () => <div data-testid="shadow-eval-section" /> }));
vi.mock("@/components/shared/advanced_date_picker", () => ({
  __esModule: true,
  default: ({ onValueChange }: { onValueChange: (value: { from?: Date; to?: Date }) => void }) => (
    <button
      type="button"
      data-testid="date-picker"
      onClick={() => onValueChange({ from: new Date(2026, 7, 1), to: new Date(2026, 7, 5) })}
    />
  ),
}));

import { useAutoRouters } from "@/app/(dashboard)/hooks/models/useModels";

import AutoRouterBenchmarksTab from "./AutoRouterBenchmarksTab";
import type {
  AutoRouterBenchmarkGroup,
  AutoRouterBenchmarksResponse,
  AutoRouterCacheStats,
} from "./autoRouterBenchmarks";
import { useAutoRouterBenchmarks } from "./useAutoRouterBenchmarks";

type HookResult = ReturnType<typeof useAutoRouterBenchmarks>;

const mockAutoRouters = (deployments: AutoRouterDeployment[] = []) => {
  vi.mocked(useAutoRouters).mockReturnValue({ data: deployments } as unknown as ReturnType<typeof useAutoRouters>);
};

const mockHook = (result: { data?: AutoRouterBenchmarksResponse; isPending?: boolean; error?: Error }) => {
  vi.mocked(useAutoRouterBenchmarks).mockReturnValue({
    data: result.data,
    isPending: result.isPending ?? false,
    error: result.error ?? null,
  } as unknown as HookResult);
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

const zeroBucket = { turns: 0, hits: 0, hit_rate_pct: 0 };

const zeroCache: AutoRouterCacheStats = {
  coverage_pct: 0,
  hit_rate_pct: 0,
  same_model: zeroBucket,
  first_visit: zeroBucket,
  return_to_tier: zeroBucket,
  unordered_turns: 0,
  return_misses_expired: 0,
  return_misses_within_ttl: 0,
  return_misses_unknown: 0,
  ttl_5m_turns: 0,
  ttl_1h_turns: 0,
};

const zeroTotals: Totals = {
  sessions: 0,
  turns: 0,
  avg_turns_per_session: 0,
  avg_session_seconds: 0,
  avg_tokens_per_session: 0,
  spend: 0,
  saved_spend: 0,
  baseline_spend: 0,
  saved_pct: 0,
  saved_per_session: 0,
  cache: zeroCache,
};

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

const renderTab = () => {
  const dateValue = { from: new Date(2026, 6, 6), to: new Date(2026, 7, 5) };
  const onDateChange = vi.fn();
  const activity = {
    dateValue,
    onDateChange,
    results: [],
    loading: false,
    isFetchingMore: false,
    progress: { currentPage: 1, totalPages: 1 },
    cancelled: false,
    cancel: vi.fn(),
  };
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return {
    dateValue,
    onDateChange,
    ...render(
      <QueryClientProvider client={queryClient}>
        <AutoRouterBenchmarksTab accessToken="sk-test" activity={activity} />
      </QueryClientProvider>,
    ),
  };
};

describe("AutoRouterBenchmarksTab", () => {
  beforeEach(() => {
    mockAutoRouters();
  });

  it("leads with total estimated savings, before the four session-shape metrics", () => {
    mockHook({ data: response([group(), group({ router_name: "gpt-auto" })]) });
    renderTab();

    const labels = screen
      .getAllByText(
        /Total estimated savings|Avg saved per session|Avg turns per session|Avg session length|Avg tokens per session/,
      )
      .map((node) => node.textContent);
    expect(labels).toEqual([
      "Total estimated savings",
      "Avg saved per session",
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
    expect(screen.getByText("Estimated spend at highest-tier model")).toBeInTheDocument();
    expect(screen.getByText("$2,534.45")).toBeInTheDocument();
    expect(screen.getByText("32.7")).toBeInTheDocument();
    expect(screen.getByText("2.1h")).toBeInTheDocument();
    expect(screen.getByText("5.3M")).toBeInTheDocument();
  });

  it("pairs the savings with the session count it was earned over, in its own tile", () => {
    mockHook({ data: response([group(), group({ router_name: "gpt-auto" })]) });
    renderTab();

    const tile = screen.getByText("Avg saved per session").closest('[data-slot="card"]');
    if (!tile) throw new Error("expected avg saved per session to render as a metric tile");

    expect(within(tile).getByText("$23.13")).toBeInTheDocument();
    expect(within(tile).getByText("· 94 sessions")).toBeInTheDocument();
  });

  it("exposes each spend row as a term and its value, not as loose text", () => {
    mockHook({ data: response([group()]) });
    renderTab();

    const terms = screen.getAllByRole("term").map((node) => node.textContent);
    const values = screen.getAllByRole("definition").map((node) => node.textContent);
    expect(terms).toEqual(["Actual auto-router spend", "Estimated spend at highest-tier model"]);
    expect(values).toEqual(["$359.86", "$2,534.45"]);
  });

  it("lets both hero columns shrink below their content so a large total cannot clip", () => {
    const huge = totals({ saved_spend: 123_456_789_012.34 });
    mockHook({ data: response([group(huge)], huge) });
    renderTab();

    const figure = screen.getByText("$123,456,789,012.34");
    const grid = figure.closest('[data-slot="card"]')?.firstElementChild;
    expect(grid).toHaveClass("md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]");
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
    expect(screen.getByRole("img", { name: "Share of turns by bucket" })).not.toHaveClass("bg-muted");
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

  it("renders the full dashboard with zeroed stats when the window has no sessions", () => {
    mockHook({ data: response([], zeroTotals) });
    renderTab();

    expect(screen.getByText("Total estimated savings")).toBeInTheDocument();
    expect(screen.getAllByText("$0.00")).toHaveLength(4);
    expect(screen.getByText("· 0 sessions")).toBeInTheDocument();
    expect(screen.getByText("0s")).toBeInTheDocument();
    expect(screen.getByText(/turns measured/)).toBeInTheDocument();
    expect(screen.getAllByText("0.0%").length).toBeGreaterThan(0);
    expect(screen.getByRole("img", { name: "Share of turns by bucket" })).toHaveClass("bg-muted");
  });

  it("shows the savings delta as an unsigned zero when nothing was saved", () => {
    mockHook({ data: response([], zeroTotals) });
    renderTab();

    expect(screen.getByText("0%")).toBeInTheDocument();
    expect(screen.queryByText("-0%")).not.toBeInTheDocument();
  });

  it("queries the shared picker's range and pushes picker changes back to the shared state", () => {
    mockHook({ data: response([group()]) });
    const { dateValue, onDateChange } = renderTab();

    expect(vi.mocked(useAutoRouterBenchmarks)).toHaveBeenCalledWith("sk-test", dateValue);
    expect(screen.getByText("Jul 6 – Aug 5 (UTC)")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("date-picker"));
    expect(onDateChange).toHaveBeenCalledWith({ from: new Date(2026, 7, 1), to: new Date(2026, 7, 5) });
  });

  it("shows usage by default and mounts shadow evals only when its sub-tab is selected", () => {
    mockHook({ data: response([group()]) });
    renderTab();

    expect(screen.getByRole("tab", { name: "Usage" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Total estimated savings")).toBeInTheDocument();
    expect(screen.queryByTestId("shadow-eval-section")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Shadow Evals" }));
    expect(screen.getByRole("tab", { name: "Shadow Evals" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("shadow-eval-section")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Usage" }));
    expect(screen.getByText("Total estimated savings")).toBeInTheDocument();
    expect(screen.getByTestId("shadow-eval-section")).toBeInTheDocument();
  });

  it("keeps the shadow evals sub-tab reachable while the usage body is in its error state", () => {
    mockHook({ error: new ApiError("boom", 500, {}) });
    renderTab();

    expect(screen.getByText("Auto-router usage is unavailable right now")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Shadow Evals" }));
    expect(screen.getByTestId("shadow-eval-section")).toBeInTheDocument();
  });

  it("keeps the range picker reachable while a window has no sessions", () => {
    mockHook({ data: response([], zeroTotals) });
    renderTab();

    expect(screen.getByTestId("date-picker")).toBeInTheDocument();
    expect(screen.getByText("All auto-routers")).toBeInTheDocument();
  });
});
