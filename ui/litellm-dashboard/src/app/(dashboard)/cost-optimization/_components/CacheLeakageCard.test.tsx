import { render } from "@testing-library/react";
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

const renderWith = (results: DailyData[]) =>
  render(
    <CacheLeakageCard
      activity={{
        dateValue: {},
        onDateChange: vi.fn(),
        results,
        loading: false,
        isFetchingMore: false,
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
      "Input tokens in the selected range that were neither read from nor written to the prompt cache",
      "Share of this key's total input tokens that were served from the prompt cache",
      "Dollars this key actually saved because cached input was billed at the discounted cache-read rate",
      "Approximate dollars this key could still save if its uncached input had hit the cache at the portfolio's realized discount",
    ].forEach((info) => expect(getByLabelText(info)).toBeInTheDocument());
  });

  it("shows an empty state when no key used tokens in the range", () => {
    const { getByText, queryByRole } = renderWith([dayWithKeys("2026-07-12", {})]);

    expect(getByText("No key usage in this range.")).toBeInTheDocument();
    expect(queryByRole("table")).not.toBeInTheDocument();
  });
});
