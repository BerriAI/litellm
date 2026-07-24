import { render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HourlySavingsResponse, ToolSpendResponse } from "@/components/networking";

import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";

const mockGetToolSpend = vi.fn();
const mockGetHourlySavings = vi.fn();

vi.mock("@/components/networking", () => ({
  getToolSpend: (...args: unknown[]) => mockGetToolSpend(...args),
  getHourlySavings: (...args: unknown[]) => mockGetHourlySavings(...args),
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
  CustomLegend: ({ categories }: { categories: readonly string[] }) => (
    <div data-testid="chart-legend">{categories.join(",")}</div>
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

interface RenderOptions {
  toolSpend?: ToolSpendResponse;
  from?: Date;
  to?: Date;
  canViewGlobalSavings?: boolean;
}

const renderWith = (results: DailyData[], options: RenderOptions = {}) => {
  const {
    toolSpend = emptyToolSpend,
    from = new Date(2026, 6, 1),
    to = new Date(2026, 6, 14),
    canViewGlobalSavings = true,
  } = options;
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
        canViewGlobalSavings,
      }}
    />,
  );
};

const hourlyResponse = (
  buckets: HourlySavingsResponse["buckets"],
  spendLogsDisabled = false,
): HourlySavingsResponse => ({
  buckets,
  start_date: "2026-07-23",
  end_date: "2026-07-23",
  timezone: "UTC",
  spend_logs_disabled: spendLogsDisabled,
});

const fullDayOfBuckets = (): HourlySavingsResponse["buckets"] =>
  Array.from({ length: 24 }, (_unused, hour) => ({
    bucket_start: `2026-07-23T${String(hour).padStart(2, "0")}:00`,
    compression_savings_spend: hour === 9 ? 0.5 : 0,
    prompt_caching_savings_spend: hour === 14 ? 2.25 : 0,
  }));

const readSeries = (element: HTMLElement) => JSON.parse(element.getAttribute("data-series") ?? "[]");

describe("UsageTab", () => {
  beforeEach(() => {
    mockGetHourlySavings.mockReset();
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

  it("opens on a running total, so each point is everything saved so far", () => {
    const { getByTestId } = renderWith(twoDays());

    const series = readSeries(getByTestId("area-chart"));
    expect(series).toHaveLength(2);
    expect(series[0]).toMatchObject({ Compression: 0.04, "Prompt caching": 0.006 });
    expect(series[1].Compression).toBeCloseTo(0.14, 5);
    expect(series[1]["Prompt caching"]).toBeCloseTo(0.016, 5);
  });

  it("drops back to the raw per-interval readings on the other tab", async () => {
    const { getByRole, getByTestId } = renderWith(twoDays());

    await userEvent.click(getByRole("tab", { name: "Per day" }));

    const series = readSeries(getByTestId("area-chart"));
    expect(series[0]).toMatchObject({ Compression: 0.04, "Prompt caching": 0.006 });
    expect(series[1]).toMatchObject({ Compression: 0.1, "Prompt caching": 0.01 });
  });

  it("names the interval tab after the granularity actually on screen", async () => {
    mockGetHourlySavings.mockResolvedValue(hourlyResponse(fullDayOfBuckets()));
    const oneDay = new Date(2026, 6, 23);
    const { getByRole, findByRole } = renderWith([], { from: oneDay, to: oneDay });

    expect(await findByRole("tab", { name: "Per hour" })).toBeInTheDocument();
    expect(() => getByRole("tab", { name: "Per day" })).toThrow();
  });

  it("says what the line means and over what range", async () => {
    const { getByText, getByRole } = renderWith(twoDays());

    expect(getByText("Running total saved \u00b7 Jul 1 \u2013 Jul 14")).toBeInTheDocument();
    await userEvent.click(getByRole("tab", { name: "Per day" }));
    expect(getByText("Saved per day \u00b7 Jul 1 \u2013 Jul 14")).toBeInTheDocument();
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
      total_spend: 5.0,
      start_date: "2026-07-12",
      end_date: "2026-07-12",
    };
    const { findAllByTestId } = renderWith([day("2026-07-12", {})], { toolSpend });

    const bars = await findAllByTestId("bar-chart");
    const series = JSON.parse(bars[0].getAttribute("data-series") ?? "[]");
    expect(series[0]).toMatchObject({ tool_name: "search", spend: 4.0 });
  });
  it("charts a single day by hour instead of collapsing it to one point", async () => {
    mockGetHourlySavings.mockResolvedValue(hourlyResponse(fullDayOfBuckets()));
    const oneDay = new Date(2026, 6, 23);
    const { getByTestId } = renderWith([day("2026-07-23", { prompt_caching_savings_spend: 2.25 })], {
      from: oneDay,
      to: oneDay,
    });

    await waitFor(() => expect(readSeries(getByTestId("area-chart"))).toHaveLength(24));
    const series = readSeries(getByTestId("area-chart"));
    expect(series[0]).toMatchObject({ date: "12am", Compression: 0, "Prompt caching": 0 });
    expect(series[9]).toMatchObject({ date: "9am", Compression: 0.5 });
    expect(series[14]).toMatchObject({ date: "2pm", "Prompt caching": 2.25 });
  });

  it("asks for the day on the viewer's own clock, by IANA zone so DST resolves per date", async () => {
    mockGetHourlySavings.mockResolvedValue(hourlyResponse(fullDayOfBuckets()));
    const oneDay = new Date(2026, 6, 23, 22, 45);
    renderWith([], { from: oneDay, to: oneDay });

    await waitFor(() => expect(mockGetHourlySavings).toHaveBeenCalled());
    expect(mockGetHourlySavings).toHaveBeenCalledWith(
      "test-token",
      "2026-07-23",
      "2026-07-23",
      Intl.DateTimeFormat().resolvedOptions().timeZone,
    );
  });

  it("labels hours with their date once the range covers more than one day", async () => {
    mockGetHourlySavings.mockResolvedValue(
      hourlyResponse([
        { bucket_start: "2026-07-23T09:00", compression_savings_spend: 1, prompt_caching_savings_spend: 0 },
        { bucket_start: "2026-07-24T09:00", compression_savings_spend: 2, prompt_caching_savings_spend: 0 },
      ]),
    );
    const { getByTestId } = renderWith([], { from: new Date(2026, 6, 23), to: new Date(2026, 6, 24) });

    await waitFor(() => expect(readSeries(getByTestId("area-chart"))).toHaveLength(2));
    expect(readSeries(getByTestId("area-chart")).map((point: { date: string }) => point.date)).toEqual([
      "7/23 9am",
      "7/24 9am",
    ]);
  });

  it("leaves long ranges on the daily rollup", async () => {
    const { getByTestId } = renderWith([
      day("2026-07-12", { compression_savings_spend: 0.04 }),
      day("2026-07-13", { compression_savings_spend: 0.1 }),
    ]);

    await waitFor(() => expect(mockGetToolSpend).toHaveBeenCalled());
    expect(mockGetHourlySavings).not.toHaveBeenCalled();
    expect(readSeries(getByTestId("area-chart"))).toHaveLength(2);
  });

  it("does not ask for hourly savings when the caller cannot read deployment-wide data", async () => {
    // e.g. an org admin: the endpoint refuses anything below proxy admin, so
    // firing the request would only 401 before the daily rollup fallback.
    const oneDay = new Date(2026, 6, 23);
    const { getByTestId } = renderWith([day("2026-07-23", { compression_savings_spend: 0.04 })], {
      from: oneDay,
      to: oneDay,
      canViewGlobalSavings: false,
    });

    await waitFor(() => expect(mockGetToolSpend).toHaveBeenCalled());
    expect(mockGetHourlySavings).not.toHaveBeenCalled();
    expect(readSeries(getByTestId("area-chart"))).toHaveLength(1);
  });

  it("falls back to the daily rollup when spend logs are turned off", async () => {
    mockGetHourlySavings.mockResolvedValue(hourlyResponse(fullDayOfBuckets(), true));
    const oneDay = new Date(2026, 6, 23);
    const { getByTestId } = renderWith([day("2026-07-23", { compression_savings_spend: 0.04 })], {
      from: oneDay,
      to: oneDay,
    });

    await waitFor(() => expect(mockGetHourlySavings).toHaveBeenCalled());
    const series = readSeries(getByTestId("area-chart"));
    expect(series).toHaveLength(1);
    expect(series[0]).toMatchObject({ Compression: 0.04 });
  });

  it("falls back to the daily rollup when the hourly request fails", async () => {
    mockGetHourlySavings.mockRejectedValue(new Error("boom"));
    const oneDay = new Date(2026, 6, 23);
    const { getByTestId } = renderWith([day("2026-07-23", { compression_savings_spend: 0.04 })], {
      from: oneDay,
      to: oneDay,
    });

    await waitFor(() => expect(mockGetHourlySavings).toHaveBeenCalled());
    const series = readSeries(getByTestId("area-chart"));
    expect(series).toHaveLength(1);
    expect(series[0]).toMatchObject({ Compression: 0.04 });
  });
});
