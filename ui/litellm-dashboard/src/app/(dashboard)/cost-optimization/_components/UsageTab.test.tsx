import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ToolSpendResponse } from "@/components/networking";

import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";

const mockGetToolSpend = vi.fn();

vi.mock("@/components/networking", () => ({
  getToolSpend: (...args: unknown[]) => mockGetToolSpend(...args),
}));

vi.mock("@/components/shared/advanced_date_picker", () => ({
  __esModule: true,
  default: () => <div data-testid="date-picker" />,
}));

vi.mock("@/components/shared/charts", () => ({
  AreaChart: ({ data, categories }: { data: unknown; categories: string[] }) => (
    <div data-testid="area-chart" data-categories={categories.join(",")} data-series={JSON.stringify(data)} />
  ),
  DonutChart: ({ data, label }: { data: unknown; label: string }) => (
    <div data-testid="donut-chart" data-label={label} data-slices={JSON.stringify(data)} />
  ),
  BarChart: ({ data, categories }: { data: unknown; categories: string[] }) => (
    <div data-testid="bar-chart" data-categories={categories.join(",")} data-series={JSON.stringify(data)} />
  ),
  DEFAULT_COLOR_CYCLE: ["emerald", "blue", "violet", "amber"],
}));

import UsageTab from "./UsageTab";

const emptyToolSpend: ToolSpendResponse = { by_tool: [], daily: [], total_spend: 0, start_date: null, end_date: null };

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

const day = (date: string, metrics: Partial<SpendMetrics>): DailyData => ({
  date,
  metrics: baseMetrics(metrics),
  breakdown: {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
  },
});

const renderWith = (results: DailyData[], toolSpend = emptyToolSpend) => {
  mockGetToolSpend.mockResolvedValue(toolSpend);
  return render(
    <UsageTab
      accessToken="test-token"
      activity={{
        dateValue: { from: new Date("2026-07-01"), to: new Date("2026-07-14") },
        onDateChange: vi.fn(),
        results,
        loading: false,
        isFetchingMore: false,
      }}
    />,
  );
};

describe("UsageTab", () => {
  it("sums compression and caching dollars across days into the summary cards", () => {
    const { getByText } = renderWith([
      day("2026-07-12", {
        compression_savings_spend: 0.04,
        prompt_caching_savings_spend: 0.006,
        compression_saved_tokens: 40000,
      }),
      day("2026-07-13", {
        compression_savings_spend: 0.1,
        prompt_caching_savings_spend: 0.01,
        compression_saved_tokens: 100000,
      }),
    ]);

    expect(getByText("$0.1560")).toBeInTheDocument();
    expect(getByText("$0.1400")).toBeInTheDocument();
    expect(getByText("$0.0160")).toBeInTheDocument();
    expect(getByText("140,000 tokens compressed")).toBeInTheDocument();
  });

  it("builds a per-day time series and per-driver donut from the daily rows", () => {
    const { getByTestId } = renderWith([
      day("2026-07-12", { compression_savings_spend: 0.04, prompt_caching_savings_spend: 0.006 }),
      day("2026-07-13", { compression_savings_spend: 0.1, prompt_caching_savings_spend: 0.01 }),
    ]);

    const series = JSON.parse(getByTestId("area-chart").getAttribute("data-series") ?? "[]");
    expect(series).toHaveLength(2);
    expect(series[0]).toMatchObject({ Compression: 0.04, "Prompt caching": 0.006 });
    expect(series[1]).toMatchObject({ Compression: 0.1, "Prompt caching": 0.01 });

    const slices = JSON.parse(getByTestId("donut-chart").getAttribute("data-slices") ?? "[]");
    expect(slices).toEqual([
      { driver: "Compression", usd: expect.closeTo(0.14, 5) },
      { driver: "Prompt caching", usd: expect.closeTo(0.016, 5) },
    ]);
  });

  it("omits a driver slice when that driver has no savings", () => {
    const { getByTestId } = renderWith([day("2026-07-12", { compression_savings_spend: 0.04 })]);

    const slices = JSON.parse(getByTestId("donut-chart").getAttribute("data-slices") ?? "[]");
    expect(slices).toEqual([{ driver: "Compression", usd: expect.closeTo(0.04, 5) }]);
  });

  it("renders spend-by-tool bars from the tool spend endpoint", async () => {
    const toolSpend = {
      by_tool: [
        { tool_name: "search", spend: 4.0, call_count: 3, total_tokens: 150 },
        { tool_name: "read_file", spend: 1.0, call_count: 2, total_tokens: 50 },
      ],
      daily: [{ date: "2026-07-12", tool_name: "search", spend: 4.0, call_count: 3 }],
      total_spend: 5.0,
      start_date: "2026-07-12",
      end_date: "2026-07-12",
    };
    const { findAllByTestId } = renderWith([day("2026-07-12", {})], toolSpend);

    const bars = await findAllByTestId("bar-chart");
    const series = JSON.parse(bars[0].getAttribute("data-series") ?? "[]");
    expect(series[0]).toMatchObject({ tool_name: "search", spend: 4.0 });
  });
});
