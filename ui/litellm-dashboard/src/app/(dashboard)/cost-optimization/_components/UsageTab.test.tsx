import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
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
  BarChart: ({
    data,
    categories,
    colors,
    showLegend,
  }: {
    data: unknown;
    categories: string[];
    colors?: readonly string[];
    showLegend?: boolean;
  }) => (
    <div
      data-testid="bar-chart"
      data-categories={categories.join(",")}
      data-colors={(colors ?? []).join(",")}
      data-show-legend={String(showLegend ?? true)}
      data-series={JSON.stringify(data)}
    />
  ),
  CustomLegend: ({ categories }: { categories: readonly string[] }) => (
    <div data-testid="chart-legend">{categories.join(",")}</div>
  ),
  SEQUENTIAL_COLOR_RAMP: ["indigo", "blue", "sky", "cyan"],
}));

import UsageTab from "./UsageTab";

const emptyToolSpend: ToolSpendResponse = { by_tool: [], daily: [], start_date: null, end_date: null };

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

interface RenderOptions {
  toolSpend?: ToolSpendResponse;
  from?: Date;
  to?: Date;
}

const renderWith = (results: DailyData[], options: RenderOptions = {}) => {
  const { toolSpend = emptyToolSpend, from = new Date(2026, 6, 1), to = new Date(2026, 6, 14) } = options;
  mockGetToolSpend.mockResolvedValue(toolSpend);
  return render(
    <UsageTab
      accessToken="test-token"
      activity={{
        dateValue: { from, to },
        onDateChange: vi.fn(),
        results,
        loading: false,
        isFetchingMore: false,
      }}
    />,
  );
};

const readSeries = (element: HTMLElement) => JSON.parse(element.getAttribute("data-series") ?? "[]");

describe("UsageTab", () => {
  beforeEach(() => {
    mockGetToolSpend.mockReset();
  });

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

  const twoDays = () => [
    day("2026-07-12", { compression_savings_spend: 0.04, prompt_caching_savings_spend: 0.006 }),
    day("2026-07-13", { compression_savings_spend: 0.1, prompt_caching_savings_spend: 0.01 }),
  ];

  it("opens on a running total anchored at $0 at the start of the range", () => {
    const { getByTestId } = renderWith(twoDays());

    // Cumulative prepends a synthetic $0 point at the range start (Jul 1) so the
    // line rises from zero rather than floating; the daily running totals follow.
    const series = readSeries(getByTestId("area-chart"));
    expect(series).toHaveLength(3);
    expect(series[0]).toMatchObject({ date: "Jul 1", Compression: 0, "Prompt caching": 0 });
    expect(series[1]).toMatchObject({ Compression: 0.04, "Prompt caching": 0.006 });
    expect(series[2].Compression).toBeCloseTo(0.14, 5);
    expect(series[2]["Prompt caching"]).toBeCloseTo(0.016, 5);
  });

  it("rises from $0 to the day's cumulative total for a single-day range", () => {
    // The original complaint: a one-day range plotted a single floating dot. The
    // synthetic start anchor gives the line a zero origin to climb from.
    const oneDay = new Date(2026, 6, 24);
    const { getByTestId } = renderWith(
      [day("2026-07-24", { compression_savings_spend: 0.2, prompt_caching_savings_spend: 0.05 })],
      { from: oneDay, to: oneDay },
    );

    const series = readSeries(getByTestId("area-chart"));
    expect(series).toHaveLength(2);
    expect(series[0]).toMatchObject({ date: "Jul 24", Compression: 0, "Prompt caching": 0 });
    expect(series[1]).toMatchObject({ date: "Jul 24", Compression: 0.2, "Prompt caching": 0.05 });
  });

  it("plots the daily series oldest first even though the rollup arrives newest first", async () => {
    // The daily activity endpoint returns days newest first; the chart must
    // still read left to right in time, and the running total must climb toward
    // the newest day, not fall away from it.
    const newestFirst = [
      day("2026-07-13", { prompt_caching_savings_spend: 0.1 }),
      day("2026-07-12", { prompt_caching_savings_spend: 0.04 }),
    ];
    const { getByTestId, getByRole } = renderWith(newestFirst);

    // The $0 anchor leads, then the days climb oldest to newest.
    const cumulative = readSeries(getByTestId("area-chart"));
    expect(cumulative.map((p: { date: string }) => p.date)).toEqual(["Jul 1", "Jul 12", "Jul 13"]);
    expect(cumulative[1]["Prompt caching"]).toBeCloseTo(0.04, 5);
    expect(cumulative[2]["Prompt caching"]).toBeCloseTo(0.14, 5);
    expect(cumulative[2]["Prompt caching"]).toBeGreaterThan(cumulative[1]["Prompt caching"]);

    await userEvent.click(getByRole("tab", { name: "Per day" }));
    const perDay = readSeries(getByTestId("bar-chart"));
    expect(perDay.map((p: { date: string }) => p.date)).toEqual(["Jul 12", "Jul 13"]);
  });

  it("draws bars of the raw per-interval readings on the other tab", async () => {
    const { getByRole, getByTestId, queryByTestId } = renderWith(twoDays());

    // Cumulative opens on the area line.
    expect(getByTestId("area-chart")).toBeInTheDocument();

    await userEvent.click(getByRole("tab", { name: "Per day" }));

    // Per day switches to a bar chart of the unaccumulated daily savings, with no
    // synthetic anchor prepended.
    expect(queryByTestId("area-chart")).toBeNull();
    const series = readSeries(getByTestId("bar-chart"));
    expect(series).toHaveLength(2);
    expect(series[0]).toMatchObject({ Compression: 0.04, "Prompt caching": 0.006 });
    expect(series[1]).toMatchObject({ Compression: 0.1, "Prompt caching": 0.01 });
  });

  it("says what the line means and over what range", async () => {
    const { getByText, getByRole } = renderWith(twoDays());

    expect(getByText("Running total saved · Jul 1 – Jul 14")).toBeInTheDocument();
    await userEvent.click(getByRole("tab", { name: "Per day" }));
    expect(getByText("Saved per day · Jul 1 – Jul 14")).toBeInTheDocument();
  });

  it("builds the per-driver donut from the range totals, not the running total", () => {
    const { getByTestId } = renderWith(twoDays());

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
      start_date: "2026-07-12",
      end_date: "2026-07-12",
    };
    const { findAllByTestId } = renderWith([day("2026-07-12", {})], { toolSpend });

    const bars = await findAllByTestId("bar-chart");
    const series = JSON.parse(bars[0].getAttribute("data-series") ?? "[]");
    expect(series[0]).toMatchObject({ tool_name: "search", spend: 4.0 });
  });

  it("renders the tool legend once outside the charts, with both charts sharing the tool colors", async () => {
    const toolSpend = {
      by_tool: [
        { tool_name: "search", spend: 4.0, call_count: 3, total_tokens: 150 },
        { tool_name: "read_file", spend: 1.0, call_count: 2, total_tokens: 50 },
      ],
      daily: [{ date: "2026-07-12", tool_name: "search", spend: 4.0, call_count: 3 }],
      start_date: "2026-07-12",
      end_date: "2026-07-12",
    };
    const { findAllByTestId, getAllByTestId } = renderWith([day("2026-07-12", {})], { toolSpend });

    const bars = await findAllByTestId("bar-chart");
    const [totalByTool, dailyByTool] = bars.slice(-2);
    expect(dailyByTool.getAttribute("data-show-legend")).toBe("false");
    expect(totalByTool.getAttribute("data-colors")).toBe(dailyByTool.getAttribute("data-colors"));

    const toolLegends = getAllByTestId("chart-legend").filter((legend) => legend.textContent === "search,read_file");
    expect(toolLegends).toHaveLength(1);
  });
});
