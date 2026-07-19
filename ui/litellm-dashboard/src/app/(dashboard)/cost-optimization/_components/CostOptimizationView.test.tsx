import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";

const mockUsePaginatedDailyActivity = vi.fn();

vi.mock("@/app/(dashboard)/usage/_components/hooks/usePaginatedDailyActivity", () => ({
  usePaginatedDailyActivity: (args: unknown) => mockUsePaginatedDailyActivity(args),
}));

vi.mock("@/components/networking", () => ({
  userDailyActivityCall: vi.fn(),
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
}));

import CostOptimizationView from "./CostOptimizationView";

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

const renderWith = (results: DailyData[]) => {
  mockUsePaginatedDailyActivity.mockReturnValue({ data: { results }, loading: false, isFetchingMore: false });
  return render(<CostOptimizationView accessToken="test-token" userId="u1" userRole="proxy_admin" />);
};

describe("CostOptimizationView", () => {
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

    // compression 0.14 + caching 0.016 = 0.156
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

  it("shows autorouter saved dollars and escalation rate from routed request counters", () => {
    const firstDay = {
      autorouter_savings_spend: 1.5,
      autorouter_requests: 10,
      autorouter_escalated_requests: 2,
      compression_savings_spend: 0.1,
    };
    const { getByText } = renderWith([
      day("2026-07-12", firstDay),
      day("2026-07-13", { autorouter_savings_spend: 2.5, autorouter_requests: 30, autorouter_escalated_requests: 1 }),
    ]);

    expect(getByText("$4.00")).toBeInTheDocument();
    expect(getByText("7.5%")).toBeInTheDocument();
    expect(getByText("3 of 40 routed requests asked to escalate")).toBeInTheDocument();
  });

  it("shows an em dash for escalation rate when there are no autorouter requests", () => {
    const { getByText } = renderWith([day("2026-07-12", { compression_savings_spend: 0.04 })]);

    expect(getByText("\u2014")).toBeInTheDocument();
    expect(getByText("No autorouter requests yet")).toBeInTheDocument();
  });

  it("adds an autorouter (est.) slice ahead of the realized-savings drivers", () => {
    const { getByTestId } = renderWith([
      day("2026-07-12", { autorouter_savings_spend: 4.0, compression_savings_spend: 0.1 }),
    ]);

    const slices = JSON.parse(getByTestId("donut-chart").getAttribute("data-slices") ?? "[]");
    expect(slices).toEqual([
      { driver: "Autorouter (est.)", usd: expect.closeTo(4.0, 5) },
      { driver: "Compression", usd: expect.closeTo(0.1, 5) },
    ]);
  });
});
