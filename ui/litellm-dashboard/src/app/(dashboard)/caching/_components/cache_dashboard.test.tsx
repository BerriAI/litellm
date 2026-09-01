import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders } from "../../../../../tests/test-utils";
import CacheDashboard from "./cache_dashboard";

const { useCacheActivity, cachingHealthCheckCall } = vi.hoisted(() => ({
  useCacheActivity: vi.fn(),
  cachingHealthCheckCall: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  cachingHealthCheckCall,
}));

vi.mock("@/app/(dashboard)/hooks/caching/useCacheActivity", () => ({
  useCacheActivity,
}));

const cacheActivity = {
  groups: [
    {
      call_type: "acompletion",
      api_requests: 1000,
      cache_hits: 300,
      failed_requests: 200,
      cached_completion_tokens: 12000,
      generated_completion_tokens: 48000,
    },
    {
      call_type: "aembedding",
      api_requests: 550,
      cache_hits: 100,
      failed_requests: 50,
      cached_completion_tokens: 2000,
      generated_completion_tokens: 9000,
    },
  ],
  totals: {
    api_requests: 1550,
    cache_hits: 400,
    failed_requests: 250,
    cached_completion_tokens: 14000,
    cache_hit_ratio: (400 / 2200) * 100,
  },
  filter_options: {
    key_aliases: ["my-key", "Unnamed Key"],
    models: ["gpt-5.1", "text-embedding-3-large"],
  },
  error_breakdown: [
    { call_type: "acompletion", error_code: "429", error_class: "RateLimitError", count: 150 },
    { call_type: "acompletion", error_code: "401", error_class: "AuthenticationError", count: 50 },
    { call_type: "aembedding", error_code: "500", error_class: "InternalServerError", count: 50 },
  ],
};

const renderDashboard = () =>
  renderWithProviders(
    <CacheDashboard accessToken="sk-test" token="tok" userRole="Admin" userID="u1" premiumUser={false} />,
  );

const REQUESTS_CHART_TITLE = "Cache Hits vs API Requests";
const TOKENS_CHART_TITLE = "Cached Completion Tokens vs Generated Completion Tokens";

// Anchored on each chart's own title rather than on a global card count, so
// adding cards elsewhere on the page cannot silently repoint these assertions.
const cardTitled = (title: string): HTMLElement => {
  const card = screen.getByText(title).closest('[data-slot="card"]');
  expect(card).not.toBeNull();
  return card as HTMLElement;
};

const findChartCards = async () => {
  await screen.findByText(REQUESTS_CHART_TITLE);
  await waitFor(() => {
    expect(document.querySelectorAll("path.recharts-rectangle").length).toBeGreaterThan(0);
  });
  return { requestsCard: cardTitled(REQUESTS_CHART_TITLE), tokensCard: cardTitled(TOKENS_CHART_TITLE) };
};

const barFills = (card: HTMLElement) =>
  Array.from(card.querySelectorAll(".recharts-bar")).map((bar) =>
    bar.querySelector("path.recharts-rectangle")?.getAttribute("fill"),
  );

const legendFillByCategory = (card: HTMLElement) =>
  Object.fromEntries(
    Array.from(card.querySelectorAll('.recharts-legend-wrapper [style*="background-color"]')).map((swatch) => [
      swatch.parentElement?.textContent,
      swatch.getAttribute("style")?.match(/background-color:\s*([^;]+);?/)?.[1],
    ]),
  );

describe("CacheDashboard cache analytics charts", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCacheActivity.mockReturnValue({ data: cacheActivity, refetch: vi.fn() });
  });

  it("renders both chart card titles", async () => {
    renderDashboard();

    expect(await screen.findByText("Cache Hits vs API Requests")).toBeInTheDocument();
    expect(screen.getByText("Cached Completion Tokens vs Generated Completion Tokens")).toBeInTheDocument();
  });

  it("scopes the analytics tab to the response cache, not provider prompt caching", async () => {
    renderDashboard();

    expect(await screen.findByText(/is not shown here/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "response cache" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/proxy/caching",
    );
    expect(screen.getByRole("link", { name: "prompt caching" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/completion/prompt_caching",
    );
    expect(screen.queryByText("Cached Tokens")).not.toBeInTheDocument();
    expect(screen.getAllByText("Cached Completion Tokens").length).toBeGreaterThan(0);
  });

  it("renders the requests chart with each category legend-bound to its fill and stacked in order", async () => {
    renderDashboard();
    const { requestsCard } = await findChartCards();

    expect(legendFillByCategory(requestsCard)).toEqual({
      "LLM API requests": "var(--color-sky-500, #0ea5e9)",
      "Cache hit": "var(--color-teal-500, #14b8a6)",
      "Failed requests": "var(--color-red-500, #ef4444)",
    });
    expect(barFills(requestsCard)).toEqual([
      "var(--color-sky-500, #0ea5e9)",
      "var(--color-teal-500, #14b8a6)",
      "var(--color-red-500, #ef4444)",
    ]);
  });

  it("renders the tokens chart with each category legend-bound to its fill and stacked in order", async () => {
    renderDashboard();
    const { tokensCard } = await findChartCards();

    expect(legendFillByCategory(tokensCard)).toEqual({
      "Generated Completion Tokens": "var(--color-sky-500, #0ea5e9)",
      "Cached Completion Tokens": "var(--color-teal-500, #14b8a6)",
    });
    expect(barFills(tokensCard)).toEqual(["var(--color-sky-500, #0ea5e9)", "var(--color-teal-500, #14b8a6)"]);
  });

  it("indexes bars by call_type name on the x axis", async () => {
    renderDashboard();
    const { requestsCard, tokensCard } = await findChartCards();

    for (const card of [requestsCard, tokensCard]) {
      expect(within(card).getAllByText("acompletion").length).toBeGreaterThan(0);
      expect(within(card).getAllByText("aembedding").length).toBeGreaterThan(0);
    }
  });

  it("stacks all categories into one column per call_type", async () => {
    renderDashboard();
    const { requestsCard, tokensCard } = await findChartCards();

    const expectedRects = { requests: 6, tokens: 4 };
    for (const [card, rectCount] of [
      [requestsCard, expectedRects.requests],
      [tokensCard, expectedRects.tokens],
    ] as const) {
      const rects = Array.from(card.querySelectorAll("path.recharts-rectangle"));
      expect(rects).toHaveLength(rectCount);
      const xPositions = rects.map((rect) => rect.getAttribute("d")?.split(",")[0]);
      expect(new Set(xPositions).size).toBe(2);
    }
  });

  it("renders the server-computed cache hit ratio", async () => {
    renderDashboard();

    expect(await screen.findByText("18.18%")).toBeInTheDocument();
  });

  it("passes the date range and selected filters to the activity query", () => {
    renderDashboard();

    expect(useCacheActivity).toHaveBeenCalledWith({
      startDate: expect.stringMatching(/^\d{4}-\d{2}-\d{2}$/),
      endDate: expect.stringMatching(/^\d{4}-\d{2}-\d{2}$/),
      keyAliases: [],
      models: [],
    });
  });

  it("opens the error-code drilldown for a call_type when its failed segment is clicked, and closes it again", async () => {
    renderDashboard();
    const { requestsCard } = await findChartCards();

    const redBar = Array.from(requestsCard.querySelectorAll(".recharts-bar")).find((bar) =>
      bar.querySelector("path.recharts-rectangle")?.getAttribute("fill")?.includes("red"),
    );
    expect(redBar).toBeDefined();
    expect(screen.queryByText(/Failed requests by error code/)).not.toBeInTheDocument();

    fireEvent.click(redBar!.querySelectorAll("path.recharts-rectangle")[0]);

    const drilldownCard = cardTitled("Failed requests by error code: acompletion");
    expect(within(drilldownCard).getAllByText("429").length).toBeGreaterThan(0);
    expect(within(drilldownCard).getAllByText("401").length).toBeGreaterThan(0);
    expect(within(drilldownCard).queryByText("500")).not.toBeInTheDocument();

    fireEvent.click(within(drilldownCard).getByRole("button", { name: "Close error breakdown" }));
    expect(screen.queryByText(/Failed requests by error code/)).not.toBeInTheDocument();
  });

  it("dismisses an open drilldown when refetched data no longer has failures for that call_type", async () => {
    const { rerender } = renderDashboard();
    const { requestsCard } = await findChartCards();

    const redBar = Array.from(requestsCard.querySelectorAll(".recharts-bar")).find((bar) =>
      bar.querySelector("path.recharts-rectangle")?.getAttribute("fill")?.includes("red"),
    );
    fireEvent.click(redBar!.querySelectorAll("path.recharts-rectangle")[0]);
    expect(screen.getByText("Failed requests by error code: acompletion")).toBeInTheDocument();

    useCacheActivity.mockReturnValue({
      data: {
        ...cacheActivity,
        groups: cacheActivity.groups.map((group) =>
          group.call_type === "acompletion" ? { ...group, failed_requests: 0 } : group,
        ),
        error_breakdown: cacheActivity.error_breakdown.filter((bucket) => bucket.call_type !== "acompletion"),
      },
      refetch: vi.fn(),
    });
    rerender(<CacheDashboard accessToken="sk-test" token="tok" userRole="Admin" userID="u1" premiumUser={false} />);

    expect(screen.queryByText(/Failed requests by error code/)).not.toBeInTheDocument();
  });

  it("formats y-axis ticks with compact notation", async () => {
    renderDashboard();
    const { requestsCard, tokensCard } = await findChartCards();

    const compactTicks = (card: HTMLElement) =>
      within(card)
        .getAllByText(/^\d+(\.\d+)?K$/)
        .map((tick) => tick.textContent);

    expect(compactTicks(requestsCard).length).toBeGreaterThan(0);
    expect(compactTicks(tokensCard)).toContain("60K");
  });
});
