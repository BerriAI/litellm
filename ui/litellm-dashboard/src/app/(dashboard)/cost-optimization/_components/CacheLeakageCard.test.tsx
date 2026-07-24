import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { DailyData, KeyMetricWithMetadata, SpendMetrics } from "@/components/UsagePage/types";

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

const renderWith = (results: DailyData[]) =>
  render(
    <CacheLeakageCard
      activity={{
        dateValue: {},
        onDateChange: vi.fn(),
        results,
        loading: false,
        isFetchingMore: false,
        canViewGlobalSavings: true,
      }}
    />,
  );

describe("CacheLeakageCard", () => {
  it("ranks leaking keys by uncached prompt tokens and shows cache hit ratio", () => {
    const { getByText, getByLabelText } = renderWith([
      dayWithKeys("2026-07-12", {
        "hash-caching": key("caching-key", { prompt_tokens: 1000, cache_read_input_tokens: 900 }),
        "hash-leaky": key("leaky-key", { prompt_tokens: 10000, cache_read_input_tokens: 0 }),
      }),
    ]);

    expect(getByText("leaky-key")).toBeInTheDocument();
    expect(getByText("0.0%")).toBeInTheDocument();
    expect(getByText("90.0%")).toBeInTheDocument();
    [
      "Input tokens you sent in this range that weren't served from or written to the cache",
      "Share of your input tokens that were served from the cache",
      "About how much you'd save if this uncached input used prompt caching. Estimated as uncached input tokens times the per-token discount your cached traffic already gets (realized cache savings ÷ cache-read tokens).",
    ].forEach((info) => expect(getByLabelText(info)).toBeInTheDocument());
  });

  it("sorts by the clicked column, worst cache hit rate first", () => {
    const { getAllByRole, getByText } = renderWith([
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
    ]);
    const firstDataRow = () => getAllByRole("row")[1];

    expect(firstDataRow()).toHaveTextContent("alpha");

    fireEvent.click(getByText("Cache hit rate"));
    expect(firstDataRow()).toHaveTextContent("bravo");

    fireEvent.click(getByText("Cache hit rate"));
    expect(firstDataRow()).toHaveTextContent("alpha");
  });

  it("switches to the model view and lists only Anthropic models", () => {
    const { getByText, queryByText } = renderWith([
      dayWithModels("2026-07-12", {
        "claude-sonnet-5": { prompt_tokens: 5000, cache_read_input_tokens: 0 },
        "gpt-4o": { prompt_tokens: 8000, cache_read_input_tokens: 0 },
      }),
    ]);

    fireEvent.click(getByText("By model"));

    expect(getByText("Cache leakage by model")).toBeInTheDocument();
    expect(getByText("claude-sonnet-5")).toBeInTheDocument();
    expect(queryByText("gpt-4o")).not.toBeInTheDocument();
  });

  it("shows an empty state when no key used tokens in the range", () => {
    const { getByText, queryByRole } = renderWith([dayWithKeys("2026-07-12", {})]);

    expect(getByText("No key usage in this range.")).toBeInTheDocument();
    expect(queryByRole("table")).not.toBeInTheDocument();
  });
});
