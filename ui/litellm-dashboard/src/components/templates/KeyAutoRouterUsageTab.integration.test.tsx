import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
  classifier_cost: 0.25,
  saved_spend: 8.75,
  baseline_spend: 10,
  saved_pct: 87.5,
  saved_per_session: 4.375,
  cache,
};

const benchmarks = {
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
const fetchMock = vi.fn(async (request: Request | string) => {
  const url = typeof request === "string" ? request : request.url;
  if (url.includes("/auto_router/benchmarks")) return jsonResponse(benchmarks);
  return jsonResponse(noDeployments);
});
const requestedUrls = () =>
  fetchMock.mock.calls.map(([request]) => (typeof request === "string" ? request : request.url));

describe("KeyAutoRouterUsageTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    vi.stubGlobal("fetch", fetchMock);
  });

  it("renders this key's spend, baseline, savings and per-router filter", async () => {
    const activity = {
      dateValue: { from: new Date(2025, 0, 1), to: new Date(2025, 0, 31) },
      onDateChange: vi.fn(),
    };
    renderWithProviders(<KeyAutoRouterUsageTab accessToken="test-token" keyToken="key-hash-1" activity={activity} />);

    expect(await screen.findByText("$8.75")).toBeInTheDocument();
    expect(screen.getByText("Actual auto-router spend")).toBeInTheDocument();
    expect(screen.getByText("$1.25")).toBeInTheDocument();
    expect(screen.getByText("LLM spend")).toBeInTheDocument();
    expect(screen.getByText("$1.00")).toBeInTheDocument();
    expect(screen.getByText("Classification cost")).toBeInTheDocument();
    expect(screen.getByText("$0.2500")).toBeInTheDocument();
    expect(screen.getByText("($62.50 / 1K turns)")).toBeInTheDocument();
    expect(screen.getByText("Estimated spend at highest-tier model")).toBeInTheDocument();
    expect(screen.getByText("$10.00")).toBeInTheDocument();
    expect(screen.getByText("Auto-router prompt caching")).toBeInTheDocument();
    expect(screen.getAllByText("50.0%").length).toBeGreaterThan(0);
    expect(screen.getByText("All auto-routers")).toBeInTheDocument();

    const benchmarkUrl = new URL(requestedUrls().find((url) => url.includes("/auto_router/benchmarks")) ?? "");
    expect(benchmarkUrl.searchParams.get("api_key")).toBe("key-hash-1");
    expect(benchmarkUrl.searchParams.get("start_date")).toBe("2025-01-01");
    expect(benchmarkUrl.searchParams.get("end_date")).toBe("2025-01-31");
  });
});
