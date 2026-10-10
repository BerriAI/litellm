import { render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { DailyData } from "@/components/UsagePage/types";
import HomePage from "./HomePage";
import { WHATS_NEW_ITEMS } from "./homeContent";

const auth = vi.hoisted(() => ({ isViewOnly: false }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => auth }));

const usage = vi.hoisted(() => ({ results: [] as DailyData[], loading: false }));
vi.mock("./useHomeUsage", () => ({
  HOME_USAGE_DAYS: 7,
  useHomeUsage: () => ({
    results: usage.results,
    totals: {
      spend: 501.74,
      requests: 5118,
      successful: 5075,
      failed: 43,
      tokens: 98_600_000,
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      avgCostPerRequest: 0.098,
      avgLatencyMs: 6100,
      successRate: 99.2,
    },
    loading: usage.loading,
    requestCountsPending: false,
    failed: false,
    budget: null,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/blogPosts/useBlogPosts", () => ({
  useBlogPosts: () => ({
    data: { posts: [{ title: "Post one", description: "d", date: "2026-10-09", url: "https://x/1" }] },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  }),
}));
vi.mock("@/app/(dashboard)/hooks/useDisableBlogPosts", () => ({ useDisableBlogPosts: () => false }));

const day = (date: string, model: string, tokens: number): DailyData =>
  ({
    date,
    metrics: {
      spend: 1,
      prompt_tokens: tokens,
      completion_tokens: 0,
      total_tokens: tokens,
      api_requests: 1,
      successful_requests: 1,
      failed_requests: 0,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
    breakdown: {
      models: {},
      model_groups: {
        [model]: {
          metrics: {
            spend: 1,
            prompt_tokens: tokens,
            completion_tokens: 0,
            total_tokens: tokens,
            api_requests: 1,
            successful_requests: 1,
            failed_requests: 0,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          },
          metadata: {},
          api_key_breakdown: {},
        },
      },
      providers: {},
      api_keys: {},
      entities: {},
      mcp_servers: {},
    },
  }) as unknown as DailyData;

const renderHome = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <HomePage />
    </QueryClientProvider>,
  );

describe("HomePage", () => {
  beforeEach(() => {
    auth.isViewOnly = false;
    usage.results = [day("2026-10-08", "gpt-6.1-sol", 900), day("2026-10-09", "claude-haiku-5-5", 100)];
  });

  it("shows the usage totals and links Create New Key to the key creation flow", () => {
    renderHome();
    expect(screen.getByRole("heading", { level: 1, name: "Home" })).toBeInTheDocument();
    expect(screen.getByText("$501.74")).toBeInTheDocument();
    expect(screen.getByText("5,118")).toBeInTheDocument();
    expect(screen.getByText("98.6M")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /create new key/i })).toHaveAttribute(
      "href",
      expect.stringMatching(/\/api-keys\?create=true$/),
    );
  });

  it("hides Create New Key from view-only users", () => {
    auth.isViewOnly = true;
    renderHome();
    expect(screen.queryByRole("link", { name: /create new key/i })).not.toBeInTheDocument();
  });

  it("lists the curated launches newest first with their links", () => {
    renderHome();
    const section = screen.getByTestId("home-whats-new");
    const links = within(section).getAllByRole("link");
    expect(links.map((link) => link.getAttribute("href"))).toEqual(WHATS_NEW_ITEMS.map((item) => item.href));
    const dates = WHATS_NEW_ITEMS.map((item) => item.publishedOn);
    expect(dates).toEqual([...dates].sort().reverse());
  });

  it("ranks model groups by token share on the leaderboard", () => {
    renderHome();
    const board = screen.getByTestId("home-leaderboard");
    const rows = within(board).getAllByRole("listitem");
    expect(within(rows[0]!).getByText("gpt-6.1-sol")).toBeInTheDocument();
    expect(within(rows[0]!).getByText("90.0%")).toBeInTheDocument();
    expect(within(rows[1]!).getByText("claude-haiku-5-5")).toBeInTheDocument();
  });

  it("renders blog posts in the Updates feed", () => {
    renderHome();
    const updates = screen.getByTestId("home-updates");
    expect(within(updates).getByRole("link", { name: /post one/i })).toHaveAttribute("href", "https://x/1");
  });
});
