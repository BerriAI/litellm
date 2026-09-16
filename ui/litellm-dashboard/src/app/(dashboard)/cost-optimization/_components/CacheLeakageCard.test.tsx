import { fireEvent, screen, waitFor } from "@testing-library/react";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../../tests/test-utils";

import type { DailyData, KeyMetricWithMetadata, SpendMetrics } from "@/components/UsagePage/types";
import type { DailyActivityRange } from "./useDailyActivityRange";

vi.mock("@/components/shared/advanced_date_picker", () => ({
  __esModule: true,
  default: () => <div data-testid="date-picker" />,
}));

import CacheLeakageCard from "./CacheLeakageCard";

const baseMetrics = (overrides: Partial<SpendMetrics>): SpendMetrics => ({
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  ...overrides,
});

const key = (alias: string, metrics: Partial<SpendMetrics>): KeyMetricWithMetadata => ({
  metrics: baseMetrics(metrics),
  metadata: { key_alias: alias, team_id: null },
});

const dayWithKeys = (date: string, apiKeys: Record<string, KeyMetricWithMetadata>): DailyData => ({
  date,
  metrics: baseMetrics({}),
  breakdown: {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: apiKeys,
    entities: {},
  },
});

const dayWithModels = (date: string, models: Record<string, Partial<SpendMetrics>>): DailyData => ({
  date,
  metrics: baseMetrics({}),
  breakdown: {
    models: Object.fromEntries(
      Object.entries(models).map(([name, m]) => [
        name,
        { metrics: baseMetrics(m), metadata: {}, api_key_breakdown: {} },
      ]),
    ),
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
  },
});

interface UrlOptions {
  searchParams?: string;
  onUrlUpdate?: OnUrlUpdateFunction;
}

const renderWith = (results: DailyData[], overrides: Partial<DailyActivityRange> = {}, url: UrlOptions = {}) =>
  renderWithProviders(
    <CacheLeakageCard
      activity={{
        dateValue: {},
        onDateChange: vi.fn(),
        results,
        loading: false,
        isFetchingMore: false,
        progress: { currentPage: 1, totalPages: 1 },
        cancelled: false,
        failed: false,
        cancel: vi.fn(),
        ...overrides,
      }}
    />,
    url,
  );

describe("CacheLeakageCard", () => {
  it("ranks leaking keys by uncached prompt tokens and shows cache hit ratio", () => {
    renderWith([
      dayWithKeys("2026-07-12", {
        "hash-caching": key("caching-key", { prompt_tokens: 1000, cache_read_input_tokens: 900 }),
        "hash-leaky": key("leaky-key", { prompt_tokens: 10000, cache_read_input_tokens: 0 }),
      }),
    ]);

    expect(screen.getByText("leaky-key")).toBeInTheDocument();
    expect(screen.getByText("0.0%")).toBeInTheDocument();
    expect(screen.getByText("90.0%")).toBeInTheDocument();
    [
      "Input tokens you sent in this range that weren't served from or written to the cache",
      "Share of your input tokens that were served from the cache",
      "About how much you'd save if this uncached input used prompt caching. Estimated as uncached input tokens times what your cached traffic already nets per cached token (realized cache savings, after write premiums, ÷ cache read and write tokens). Blank when caching is not currently saving anything overall.",
    ].forEach((info) => expect(screen.getByLabelText(info)).toBeInTheDocument());
  });

  const twoKeys = () => [
    dayWithKeys("2026-07-12", {
      "hash-a": key("alpha", {
        prompt_tokens: 10000,
        cache_read_input_tokens: 9000,
        prompt_caching_savings_spend: 9.0,
      }),
      "hash-b": key("bravo", {
        prompt_tokens: 500,
        cache_read_input_tokens: 50,
        prompt_caching_savings_spend: 0.05,
      }),
    }),
  ];
  const firstDataRow = () => screen.getAllByRole("row")[1];
  const lastSearchParams = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
    onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

  it("sorts by the clicked column, worst cache hit rate first", () => {
    renderWith(twoKeys());

    expect(firstDataRow()).toHaveTextContent("alpha");

    fireEvent.click(screen.getByText("Cache hit rate"));
    expect(firstDataRow()).toHaveTextContent("bravo");

    fireEvent.click(screen.getByText("Cache hit rate"));
    expect(firstDataRow()).toHaveTextContent("alpha");
  });

  it("switches to the model view and lists only Anthropic models", () => {
    renderWith([
      dayWithModels("2026-07-12", {
        "claude-sonnet-5": { prompt_tokens: 5000, cache_read_input_tokens: 0 },
        "gpt-4o": { prompt_tokens: 8000, cache_read_input_tokens: 0 },
      }),
    ]);

    fireEvent.click(screen.getByText("By model"));

    expect(screen.getByText("Cache leakage by model")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet-5")).toBeInTheDocument();
    expect(screen.queryByText("gpt-4o")).not.toBeInTheDocument();
  });

  it("applies the sort named in ?leak_sort= and ?leak_dir=", () => {
    renderWith(twoKeys(), {}, { searchParams: "?leak_sort=cacheHitRatio&leak_dir=asc" });

    expect(firstDataRow()).toHaveTextContent("bravo");
  });

  it("keeps the default sort for unknown ?leak_sort= and ?leak_dir= values", () => {
    renderWith(twoKeys(), {}, { searchParams: "?leak_sort=spend&leak_dir=sideways" });

    expect(firstDataRow()).toHaveTextContent("alpha");
  });

  it("writes the sort to the URL and drops the direction once it is back to the default", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWith(twoKeys(), {}, { onUrlUpdate });

    fireEvent.click(screen.getByText("Cache hit rate"));
    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("leak_sort")).toBe("cacheHitRatio"));
    expect(lastSearchParams(onUrlUpdate)?.get("leak_dir")).toBe("asc");

    fireEvent.click(screen.getByText("Cache hit rate"));
    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.has("leak_dir")).toBe(false));
    expect(lastSearchParams(onUrlUpdate)?.get("leak_sort")).toBe("cacheHitRatio");
    expect(firstDataRow()).toHaveTextContent("alpha");
  });

  it("opens the model view from ?leak_by=model", () => {
    renderWith(
      [
        dayWithModels("2026-07-12", {
          "claude-sonnet-5": { prompt_tokens: 5000, cache_read_input_tokens: 0 },
        }),
      ],
      {},
      { searchParams: "?leak_by=model" },
    );

    expect(screen.getByRole("tab", { name: "By model" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Cache leakage by model")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet-5")).toBeInTheDocument();
  });

  it("writes ?leak_by=model for the model view and drops it for the key view", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWith(twoKeys(), {}, { onUrlUpdate });

    fireEvent.click(screen.getByRole("tab", { name: "By model" }));
    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("leak_by")).toBe("model"));

    fireEvent.click(screen.getByRole("tab", { name: "By virtual key" }));
    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.has("leak_by")).toBe(false));
    expect(screen.getByText("alpha")).toBeInTheDocument();
  });

  it("shows an empty state when no key used tokens in the range", () => {
    renderWith([dayWithKeys("2026-07-12", {})]);

    expect(screen.getByText("No key usage in this range.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("tells the user the table is still filling in while fallback pages stream", () => {
    const day = dayWithKeys("2026-07-12", {
      "hash-leaky": key("leaky-key", { prompt_tokens: 10000, cache_read_input_tokens: 0 }),
    });
    renderWith([day], { isFetchingMore: true });

    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(
      screen.getByText("Data is still loading; rows and totals will update as the rest of the range arrives."),
    ).toBeInTheDocument();
  });

  it("keeps the streaming note off while a fresh range loads over the previous range's rows", () => {
    const day = dayWithKeys("2026-07-12", {
      "hash-leaky": key("leaky-key", { prompt_tokens: 10000, cache_read_input_tokens: 0 }),
    });
    renderWith([day], { loading: true });

    expect(
      screen.queryByText("Data is still loading; rows and totals will update as the rest of the range arrives."),
    ).not.toBeInTheDocument();
  });

  it("drops the streaming note once the range has settled", () => {
    const day = dayWithKeys("2026-07-12", {
      "hash-leaky": key("leaky-key", { prompt_tokens: 10000, cache_read_input_tokens: 0 }),
    });
    renderWith([day]);

    expect(
      screen.queryByText("Data is still loading; rows and totals will update as the rest of the range arrives."),
    ).not.toBeInTheDocument();
  });
});
