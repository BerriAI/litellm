import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AutoRouterBenchmarksTab from "@/app/(dashboard)/cost-optimization/_components/AutoRouterBenchmarksTab";
import type {
  AutoRouterBenchmarkGroup,
  AutoRouterBenchmarksResponse,
} from "@/app/(dashboard)/cost-optimization/_components/autoRouterBenchmarks";

import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import KeyAutoRouterUsageTab from "./KeyAutoRouterUsageTab";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token", userId: "admin-123", userRole: "Admin" }),
}));

const jsonResponse = (body: unknown) =>
  new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });

const cache = {
  coverage_pct: 100,
  hit_rate_pct: 50,
  same_model: { turns: 2, hits: 1, hit_rate_pct: 50 },
  first_visit: { turns: 1, hits: 0, hit_rate_pct: 0 },
  return_to_tier: { turns: 1, hits: 1, hit_rate_pct: 100 },
  unordered_turns: 0,
  return_misses_expired: 0,
  return_misses_within_ttl: 0,
  return_misses_unknown: 0,
  ttl_5m_turns: 0,
  ttl_1h_turns: 0,
};

const stats = {
  sessions: 2,
  turns: 4,
  avg_turns_per_session: 2,
  avg_session_seconds: 30,
  avg_tokens_per_session: 100,
  spend: 1.25,
  savings_estimated_turns: 4,
  savings_estimated_actual_spend: 1.25,
  savings_estimated_classifier_cost: 0.25,
  classifier_cost: 0.25,
  saved_spend: 8.75,
  baseline_spend: 10,
  saved_pct: 87.5,
  cache,
};

const benchmarks: AutoRouterBenchmarksResponse = {
  start_date: "2025-01-01",
  end_date: "2025-01-31",
  routers_in_scope: 2,
  totals: stats,
  groups: [
    { router_name: "router-one", router_type: "complexity", tier_turns: { SIMPLE: 4 }, ...stats },
    {
      router_name: "router-two",
      router_type: "complexity",
      tier_turns: { SIMPLE: 1 },
      ...stats,
      spend: 0.25,
      saved_spend: 0.75,
      baseline_spend: 1,
    },
  ],
};

const noDeployments = { data: [], total_count: 0, current_page: 1, total_pages: 1, size: 1000 };
const fetchMock = vi.fn<(request: Request | string) => Promise<Response>>();
const mockBenchmarks = (body: AutoRouterBenchmarksResponse) => {
  fetchMock.mockImplementation(async (request) => {
    const url = typeof request === "string" ? request : request.url;
    return jsonResponse(url.includes("/auto_router/benchmarks") ? body : noDeployments);
  });
};
const group = (overrides: Partial<AutoRouterBenchmarkGroup> = {}): AutoRouterBenchmarkGroup => ({
  ...stats,
  router_name: "claude-auto",
  router_type: "complexity",
  ...overrides,
});
const response = (groups: AutoRouterBenchmarkGroup[]): AutoRouterBenchmarksResponse => ({
  ...benchmarks,
  routers_in_scope: groups.length,
  groups,
});
const renderOverallTab = () => {
  const activity = {
    dateValue: { from: new Date(2025, 0, 1), to: new Date(2025, 0, 31) },
    onDateChange: vi.fn(),
    results: [],
    loading: false,
    isFetchingMore: false,
    progress: { currentPage: 1, totalPages: 1 },
    cancelled: false,
    cancel: vi.fn(),
  };
  renderWithProviders(<AutoRouterBenchmarksTab accessToken="test-token" activity={activity} />);
};
const requestedUrls = () =>
  fetchMock.mock.calls.map(([request]) => (typeof request === "string" ? request : request.url));

describe("Auto-router usage views", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockBenchmarks(benchmarks);
    testQueryClient.clear();
    vi.stubGlobal("fetch", fetchMock);
  });

  it("renders this key's spend, baseline, savings and per-router filter", async () => {
    const activity = {
      dateValue: { from: new Date(2025, 0, 1), to: new Date(2025, 0, 31) },
      onDateChange: vi.fn(),
    };
    renderWithProviders(<KeyAutoRouterUsageTab accessToken="test-token" keyToken="key-hash-1" activity={activity} />);

    const savings = within(await screen.findByRole("region", { name: "Auto-router savings" }));
    expect(savings.getByText("$8.75")).toBeInTheDocument();
    expect(savings.getByText("Actual auto-router spend")).toBeInTheDocument();
    expect(savings.getByText("$1.25")).toBeInTheDocument();
    expect(savings.getByText("LLM spend")).toBeInTheDocument();
    expect(savings.getByText("$1.00")).toBeInTheDocument();
    expect(savings.getByText("Classification cost")).toBeInTheDocument();
    expect(savings.getByText("$0.2500")).toBeInTheDocument();
    expect(savings.getByText("($62.50 / 1K turns)")).toBeInTheDocument();
    expect(savings.getByText("Estimated baseline spend")).toBeInTheDocument();
    expect(savings.getByText("$10.00")).toBeInTheDocument();
    expect(screen.getByText("Auto-router prompt caching")).toBeInTheDocument();
    expect(screen.getAllByText("50.0%").length).toBeGreaterThan(0);
    expect(screen.getByText("All auto-routers")).toBeInTheDocument();
    const summary = within(screen.getByRole("table", { name: "Router usage and savings" }));
    expect(summary.getByRole("row", { name: /router-one.*\$1\.25.*\$8\.75/ })).toBeInTheDocument();
    expect(summary.getByRole("row", { name: /router-two.*\$0\.25.*\$0\.75/ })).toBeInTheDocument();

    const benchmarkUrl = new URL(requestedUrls().find((url) => url.includes("/auto_router/benchmarks")) ?? "");
    expect(benchmarkUrl.searchParams.get("api_key")).toBe("key-hash-1");
    expect(benchmarkUrl.searchParams.get("start_date")).toBe("2025-01-01");
    expect(benchmarkUrl.searchParams.get("end_date")).toBe("2025-01-31");
  });
  it("shows per-router token costs above caching and follows the router picker", async () => {
    const standard = { total_tokens: 100_000_000, spend: 20_000, saved_spend: 4_000, saved_pct: 16.7 };
    const losing = { router_name: "gpt-auto", total_tokens: 2_000_000, spend: 15, saved_spend: -5, saved_pct: -50 };
    mockBenchmarks(response([group(standard), group(losing)]));
    renderOverallTab();

    const table = await screen.findByRole("table", { name: "Router usage and savings" });
    const rows = within(table).getAllByRole("row");
    expect(
      within(rows[1])
        .getAllByRole("cell")
        .map((cell) => cell.textContent),
    ).toEqual(["claude-auto", "100,000,000", "$20,000.00", "$200.00", "$4,000.00", "16.7%"]);
    expect(
      within(rows[2])
        .getAllByRole("cell")
        .map((cell) => cell.textContent),
    ).toEqual(["gpt-auto", "2,000,000", "$15.00", "$7.50", "-$5.00", "-50.0%"]);
    expect(
      table.compareDocumentPosition(screen.getByText("Auto-router prompt caching")) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "gpt-auto" }));
    await waitFor(() => expect(within(table).getAllByRole("row")).toHaveLength(2));
    expect(within(table).queryByText("claude-auto")).not.toBeInTheDocument();
    expect(within(table).getByText("-$5.00")).toBeInTheDocument();
  });

  it.each([null, undefined, 0])("keeps token coverage and unit costs honest for %s tokens", async (total_tokens) => {
    const untracked = { total_tokens, saved_spend: null, saved_pct: null, baseline_spend: null };
    mockBenchmarks(response([group(untracked)]));
    renderOverallTab();
    const row = within(await screen.findByRole("table", { name: "Router usage and savings" })).getAllByRole("row")[1];
    expect(
      within(row)
        .getAllByRole("cell")
        .map((cell) => cell.textContent),
    ).toEqual([
      "claude-auto",
      total_tokens === 0 ? "0" : "Unavailable",
      "$1.25",
      "Unavailable",
      "Unavailable",
      "Unavailable",
    ]);
  });

  it("shows a clear empty summary when no routers are present", async () => {
    mockBenchmarks(response([]));
    renderOverallTab();
    expect(
      within(await screen.findByRole("table", { name: "Router usage and savings" })).getByText(
        "No auto-routers in this range",
      ),
    ).toBeInTheDocument();
  });
});
