import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { endOfDay } from "date-fns";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { QueryClientProvider } from "@tanstack/react-query";
import { renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
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

vi.mock("@/components/shared/advanced_date_picker", () => ({
  __esModule: true,
  default: ({ onValueChange }: { onValueChange: (value: DateRangePickerValue) => void }) => (
    <button
      type="button"
      data-testid="date-picker"
      onClick={() => onValueChange({ from: new Date(2026, 7, 1), to: new Date(2026, 7, 5, 23, 59, 59, 999) })}
    />
  ),
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

const renderDashboard = (url: { searchParams?: string; onUrlUpdate?: OnUrlUpdateFunction } = {}) =>
  renderWithProviders(
    <CacheDashboard accessToken="sk-test" token="tok" userRole="Admin" userID="u1" premiumUser={false} />,
    url,
  );

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

const failedBarOf = (card: HTMLElement) => {
  const redBar = Array.from(card.querySelectorAll(".recharts-bar")).find((bar) =>
    bar.querySelector("path.recharts-rectangle")?.getAttribute("fill")?.includes("red"),
  );
  expect(redBar).toBeDefined();
  return redBar!.querySelectorAll("path.recharts-rectangle")[0];
};

const utcDayOf = (date: Date) => date.toISOString().slice(0, 10);

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

    expect(screen.queryByText(/Failed requests by error code/)).not.toBeInTheDocument();

    fireEvent.click(failedBarOf(requestsCard));

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

    fireEvent.click(failedBarOf(requestsCard));
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

  it("explains the Unknown bucket only when a group has no recorded endpoint", async () => {
    const { rerender } = renderDashboard();
    await screen.findByText(REQUESTS_CHART_TITLE);
    expect(screen.queryByText(/recorded no endpoint/)).not.toBeInTheDocument();

    useCacheActivity.mockReturnValue({
      data: {
        ...cacheActivity,
        groups: [
          ...cacheActivity.groups,
          {
            call_type: "Unknown",
            api_requests: 0,
            cache_hits: 0,
            failed_requests: 121000,
            cached_completion_tokens: 0,
            generated_completion_tokens: 0,
          },
        ],
      },
      refetch: vi.fn(),
    });
    rerender(<CacheDashboard accessToken="sk-test" token="tok" userRole="Admin" userID="u1" premiumUser={false} />);

    expect(
      within(cardTitled(REQUESTS_CHART_TITLE)).getByText(/Unknown groups spend logs that recorded no endpoint/),
    ).toHaveTextContent("not necessarily LLM API requests");
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

describe("CacheDashboard URL state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCacheActivity.mockReturnValue({ data: cacheActivity, refetch: vi.fn() });
  });

  it("opens the tab named in ?tab=", () => {
    renderDashboard({ searchParams: "?tab=coordination-redis" });

    expect(screen.getByRole("tab", { name: "Coordination Redis" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Cache Analytics" })).toHaveAttribute("aria-selected", "false");
  });

  it("writes ?tab= when a tab is chosen and drops it for analytics", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderDashboard({ onUrlUpdate });

    fireEvent.click(screen.getByRole("tab", { name: "Cache Health" }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("health"));
    expect(screen.getByRole("tab", { name: "Cache Health" })).toHaveAttribute("aria-selected", "true");

    fireEvent.click(screen.getByRole("tab", { name: "Cache Analytics" }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false));
  });

  it("falls back to analytics and clears an unknown ?tab= while keeping the filters", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    render(<CacheDashboard accessToken="sk-test" token="tok" userRole="Admin" userID="u1" premiumUser={false} />, {
      wrapper: ({ children }) => (
        <NuqsTestingAdapter
          searchParams="?tab=coordination&keys=my-key"
          onUrlUpdate={onUrlUpdate}
          hasMemory
          resetUrlUpdateQueueOnMount={false}
        >
          <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
        </NuqsTestingAdapter>
      ),
    });

    expect(screen.getByRole("tab", { name: "Cache Analytics" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("keys")).toBe("my-key");
  });

  it("queries the days, keys and models named in the URL and shows them as chips", () => {
    renderDashboard({
      searchParams: "?start_date=2026-01-05&end_date=2026-01-07&keys=my-key&models=gpt-5.1,text-embedding-3-large",
    });

    const expectedQuery = {
      startDate: utcDayOf(new Date(2026, 0, 5)),
      endDate: utcDayOf(endOfDay(new Date(2026, 0, 7))),
      keyAliases: ["my-key"],
      models: ["gpt-5.1", "text-embedding-3-large"],
    };
    expect(useCacheActivity).toHaveBeenLastCalledWith(expectedQuery);
    expect(screen.getByLabelText("my-key")).toBeInTheDocument();
    expect(screen.getByLabelText("gpt-5.1")).toBeInTheDocument();
    expect(screen.getByLabelText("text-embedding-3-large")).toBeInTheDocument();
  });

  it("ignores a reversed date range in the URL and keeps querying the default week", () => {
    renderDashboard({ searchParams: "?start_date=2026-01-07&end_date=2026-01-05" });

    const { startDate, endDate } = useCacheActivity.mock.calls.at(-1)![0];
    expect(startDate).not.toBe("2026-01-07");
    expect(endDate).not.toBe("2026-01-05");
    expect(startDate < endDate).toBe(true);
  });

  it("writes the picked days to ?start_date= and ?end_date= and queries them", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderDashboard({ onUrlUpdate });

    fireEvent.click(screen.getByTestId("date-picker"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("start_date")).toBe("2026-08-01"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("end_date")).toBe("2026-08-05");
    expect(useCacheActivity).toHaveBeenLastCalledWith(
      expect.objectContaining({
        startDate: utcDayOf(new Date(2026, 7, 1)),
        endDate: utcDayOf(endOfDay(new Date(2026, 7, 5))),
      }),
    );
  });

  describe("outside UTC", () => {
    afterEach(() => {
      vi.unstubAllEnvs();
    });

    it.each([
      ["west of UTC, keeping the last hours of the end day", "America/Los_Angeles", "2026-01-05", "2026-01-08"],
      ["east of UTC, keeping the first hours of the start day", "Asia/Tokyo", "2026-01-04", "2026-01-07"],
    ])("queries the UTC days that cover the local URL days %s", (_label, timeZone, startDate, endDate) => {
      vi.stubEnv("TZ", timeZone);
      renderDashboard({ searchParams: "?start_date=2026-01-05&end_date=2026-01-07" });

      expect(useCacheActivity).toHaveBeenLastCalledWith(expect.objectContaining({ startDate, endDate }));
    });
  });

  it("writes a chosen virtual key to ?keys= and removes it again", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderDashboard({ onUrlUpdate });

    await user.click(screen.getByPlaceholderText("Select Virtual Keys"));
    await user.click(await screen.findByRole("option", { name: "my-key" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("keys")).toBe("my-key"));
    expect(useCacheActivity).toHaveBeenLastCalledWith(expect.objectContaining({ keyAliases: ["my-key"] }));

    await user.click(within(screen.getByLabelText("my-key")).getByRole("button"));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("keys")).toBe(false));
  });

  it("writes a chosen model to ?models=", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderDashboard({ onUrlUpdate });

    await user.click(screen.getByPlaceholderText("Select Models"));
    await user.click(await screen.findByRole("option", { name: "gpt-5.1" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("models")).toBe("gpt-5.1"));
    expect(useCacheActivity).toHaveBeenLastCalledWith(expect.objectContaining({ models: ["gpt-5.1"] }));
  });

  it("opens the drilldown named in ?error_call_type=", () => {
    renderDashboard({ searchParams: "?error_call_type=aembedding" });

    const drilldownCard = cardTitled("Failed requests by error code: aembedding");
    expect(within(drilldownCard).getAllByText("500").length).toBeGreaterThan(0);
  });

  it.each([
    ["a call_type the data does not have", "?error_call_type=aimage_generation"],
    ["a call_type with no failures", "?error_call_type=aresponses"],
  ])("keeps the drilldown closed for %s", async (_label, searchParams) => {
    useCacheActivity.mockReturnValue({
      data: {
        ...cacheActivity,
        groups: [...cacheActivity.groups, { ...cacheActivity.groups[1], call_type: "aresponses", failed_requests: 0 }],
      },
      refetch: vi.fn(),
    });
    renderDashboard({ searchParams });
    await screen.findByText(REQUESTS_CHART_TITLE);

    expect(screen.queryByText(/Failed requests by error code/)).not.toBeInTheDocument();
  });

  it("pushes the clicked call_type to ?error_call_type= and removes it on close", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderDashboard({ onUrlUpdate });
    const { requestsCard } = await findChartCards();

    fireEvent.click(failedBarOf(requestsCard));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("error_call_type")).toBe("acompletion"));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");

    fireEvent.click(screen.getByRole("button", { name: "Close error breakdown" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("error_call_type")).toBe(false));
    expect(screen.queryByText(/Failed requests by error code/)).not.toBeInTheDocument();
  });
});
