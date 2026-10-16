import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { components } from "@/lib/http/schema";
import type { KeySpendActivityRow } from "@/components/UsagePage/dailyActivityApi";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { DailyActivityRange } from "./useDailyActivityRange";

const mockCacheLeakageKeysCall = vi.fn();

vi.mock("@/components/networking", () => ({
  cacheLeakageKeysCall: (...args: unknown[]) => mockCacheLeakageKeysCall(...args),
}));

vi.mock("@/components/shared/advanced_date_picker", () => ({
  __esModule: true,
  default: () => <div data-testid="date-picker" />,
}));

import CacheLeakageCard from "./CacheLeakageCard";

const baseMetrics = (overrides: Partial<SpendMetrics>): components["schemas"]["SpendMetrics"] => ({
  spend: 0,
  flat_cost: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  compression_saved_tokens: 0,
  compression_savings_spend: 0,
  prompt_caching_savings_spend: 0,
  gateway_injected_caching_savings_spend: 0,
  autorouter_savings_spend: 0,
  total_response_time_ms: 0,
  timed_requests: 0,
  ...overrides,
});

const keyRow = (hash: string, alias: string, metrics: Partial<SpendMetrics>): KeySpendActivityRow => ({
  api_key: hash,
  metrics: baseMetrics(metrics),
  metadata: { key_alias: alias, team_id: null },
});

const dayWithModels = (date: string, models: Record<string, Partial<SpendMetrics>>): DailyData => ({
  date,
  metrics: baseMetrics({}),
  breakdown: {
    models: {},
    model_groups: Object.fromEntries(
      Object.entries(models).map(([name, m]) => [
        name,
        { metrics: baseMetrics(m), metadata: {}, api_key_breakdown: {} },
      ]),
    ),
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
  },
});

const renderWith = (results: DailyData[], overrides: Partial<DailyActivityRange> = {}) =>
  render(
    <CacheLeakageCard
      activity={{
        dateValue: {},
        onDateChange: vi.fn(),
        results,
        metadata: EMPTY_DAILY_ACTIVITY_METADATA,
        loading: false,
        failed: false,
        scope: {
          accessToken: "test-token",
          startTime: new Date(2025, 0, 1),
          endTime: new Date(2025, 0, 31),
          userId: null,
          apiKey: null,
        },
        ...overrides,
      }}
    />,
  );

describe("CacheLeakageCard", () => {
  beforeEach(() => {
    mockCacheLeakageKeysCall.mockReset();
    mockCacheLeakageKeysCall.mockResolvedValue({ api_keys: [] });
  });

  it("ranks leaking keys from the server-ranked key list and shows cache hit ratio", async () => {
    mockCacheLeakageKeysCall.mockResolvedValue({
      api_keys: [
        keyRow("hash-caching", "caching-key", { prompt_tokens: 1000, cache_read_input_tokens: 900 }),
        keyRow("hash-leaky", "leaky-key", { prompt_tokens: 10000, cache_read_input_tokens: 0 }),
      ],
    });
    renderWith([]);

    expect(await screen.findByText("leaky-key")).toBeInTheDocument();
    expect(screen.getByText("0.0%")).toBeInTheDocument();
    expect(screen.getByText("90.0%")).toBeInTheDocument();
    [
      "Input tokens you sent in this range that weren't served from or written to the cache",
      "Share of your input tokens that were served from the cache",
      "About how much you'd save if this uncached input used prompt caching. Estimated as uncached input tokens times what your cached traffic already nets per cached token (realized cache savings, after write premiums, ÷ cache read and write tokens). Blank when caching is not currently saving anything overall.",
    ].forEach((info) => expect(screen.getByLabelText(info)).toBeInTheDocument());
  });

  it("asks the server for the key ranking under the activity scope", async () => {
    renderWith([], {
      scope: {
        accessToken: "test-token",
        startTime: new Date(2025, 0, 1),
        endTime: new Date(2025, 0, 31),
        userId: "u1",
        apiKey: "hash-1",
      },
    });

    await screen.findByText("No key usage in this range.");
    expect(mockCacheLeakageKeysCall).toHaveBeenCalledWith(
      expect.objectContaining({ entityIds: ["u1"], apiKey: "hash-1", includeCurrentUtcDay: true }),
    );
  });

  it("sorts by the clicked column, worst cache hit rate first", async () => {
    mockCacheLeakageKeysCall.mockResolvedValue({
      api_keys: [
        keyRow("hash-a", "alpha", {
          prompt_tokens: 10000,
          cache_read_input_tokens: 9000,
          prompt_caching_savings_spend: 9.0,
        }),
        keyRow("hash-b", "bravo", {
          prompt_tokens: 500,
          cache_read_input_tokens: 50,
          prompt_caching_savings_spend: 0.05,
        }),
      ],
    });
    renderWith([]);
    const firstDataRow = () => screen.getAllByRole("row")[1];

    expect(await screen.findByText("alpha")).toBeInTheDocument();
    expect(firstDataRow()).toHaveTextContent("alpha");

    fireEvent.click(screen.getByText("Cache hit rate"));
    expect(firstDataRow()).toHaveTextContent("bravo");

    fireEvent.click(screen.getByText("Cache hit rate"));
    expect(firstDataRow()).toHaveTextContent("alpha");
  });

  it("switches to the model view and lists models from every provider", async () => {
    renderWith([
      dayWithModels("2026-07-12", {
        "claude-sonnet-5": { prompt_tokens: 5000, cache_read_input_tokens: 0 },
        "vertex_ai/gemini-2.5-pro": { prompt_tokens: 8000, cache_read_input_tokens: 2000 },
      }),
    ]);

    fireEvent.click(await screen.findByText("By model"));

    expect(screen.getByText("Cache leakage by model")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet-5")).toBeInTheDocument();
    expect(screen.getByText("vertex_ai/gemini-2.5-pro")).toBeInTheDocument();
  });

  it("shows an empty state when no key used tokens in the range", async () => {
    renderWith([]);

    expect(await screen.findByText("No key usage in this range.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("reports a load failure instead of claiming the range is empty", async () => {
    mockCacheLeakageKeysCall.mockRejectedValue(new Error("route unavailable"));
    renderWith([]);

    expect(await screen.findByText("Could not load key usage for this range.")).toBeInTheDocument();
    expect(screen.queryByText("No key usage in this range.")).not.toBeInTheDocument();
  });

  it("shows a loading state while the key ranking is in flight", () => {
    mockCacheLeakageKeysCall.mockReturnValue(new Promise(() => {}));
    renderWith([]);

    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });
});
