import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders } from "../../../../../tests/test-utils";
import CacheDashboard from "./cache_dashboard";

const { adminGlobalCacheActivity, cachingHealthCheckCall } = vi.hoisted(() => ({
  adminGlobalCacheActivity: vi.fn(),
  cachingHealthCheckCall: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  adminGlobalCacheActivity,
  cachingHealthCheckCall,
}));

const cacheActivity = [
  {
    api_key: "sk-1",
    model: "gpt-5.1",
    call_type: "acompletion",
    total_rows: 1500,
    cache_hit_true_rows: 300,
    cached_completion_tokens: 12000,
    generated_completion_tokens: 48000,
  },
  {
    api_key: "sk-2",
    model: "text-embedding-3-large",
    call_type: "aembedding",
    total_rows: 700,
    cache_hit_true_rows: 100,
    cached_completion_tokens: 2000,
    generated_completion_tokens: 9000,
  },
];

const renderDashboard = () =>
  renderWithProviders(
    <CacheDashboard accessToken="sk-test" token="tok" userRole="Admin" userID="u1" premiumUser={false} />,
  );

const findChartCards = async () => {
  await screen.findByText("Cache Hits vs API Requests");
  await waitFor(() => {
    expect(document.querySelectorAll("path.recharts-rectangle").length).toBeGreaterThan(0);
  });
  const cards = Array.from(document.querySelectorAll('[data-slot="card"]'));
  expect(cards).toHaveLength(2);
  return { requestsCard: cards[0] as HTMLElement, tokensCard: cards[1] as HTMLElement };
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
    adminGlobalCacheActivity.mockResolvedValue(cacheActivity);
  });

  it("renders both chart card titles", async () => {
    renderDashboard();

    expect(await screen.findByText("Cache Hits vs API Requests")).toBeInTheDocument();
    expect(screen.getByText("Cached Completion Tokens vs Generated Completion Tokens")).toBeInTheDocument();
  });

  it("renders the requests chart with each category legend-bound to its fill and stacked in order", async () => {
    renderDashboard();
    const { requestsCard } = await findChartCards();

    expect(legendFillByCategory(requestsCard)).toEqual({
      "LLM API requests": "var(--color-sky-500, #0ea5e9)",
      "Cache hit": "var(--color-teal-500, #14b8a6)",
    });
    expect(barFills(requestsCard)).toEqual(["var(--color-sky-500, #0ea5e9)", "var(--color-teal-500, #14b8a6)"]);
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

  it("stacks the two categories into one column per call_type", async () => {
    renderDashboard();
    const { requestsCard, tokensCard } = await findChartCards();

    for (const card of [requestsCard, tokensCard]) {
      const rects = Array.from(card.querySelectorAll("path.recharts-rectangle"));
      expect(rects).toHaveLength(4);
      const xPositions = rects.map((rect) => rect.getAttribute("d")?.split(",")[0]);
      expect(new Set(xPositions).size).toBe(2);
    }
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
