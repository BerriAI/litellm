import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";

const mockUsePaginatedDailyActivity = vi.fn();
const mockUseQuery = vi.fn();
const mockUiSpendLogsCall = vi.fn();

vi.mock("@tanstack/react-query", () => ({
  useQuery: (args: unknown) => mockUseQuery(args),
}));

vi.mock("@/app/(dashboard)/usage/_components/hooks/usePaginatedDailyActivity", () => ({
  usePaginatedDailyActivity: (args: unknown) => mockUsePaginatedDailyActivity(args),
}));

vi.mock("@/components/networking", () => ({
  userDailyActivityCall: vi.fn(),
  getCostOptimizationUsageLogs: vi.fn(),
  uiSpendLogsCall: (args: unknown) => mockUiSpendLogsCall(args),
}));

vi.mock("@/components/view_logs/LogDetailsDrawer", () => ({
  LogDetailsDrawer: ({ open, logEntry }: { open: boolean; logEntry: { request_id: string } | null }) =>
    open && logEntry ? <div data-testid="log-details-drawer">{logEntry.request_id}</div> : null,
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

const detailLog = {
  request_id: "req-123456789",
  api_key: "key",
  team_id: "team",
  model: "test-model",
  model_id: "model-id",
  call_type: "completion",
  spend: 0.02,
  total_tokens: 150,
  prompt_tokens: 100,
  completion_tokens: 50,
  startTime: "2026-07-13T12:00:00Z",
  endTime: "2026-07-13T12:00:01Z",
  messages: [],
  response: {},
  cache_hit: "",
  metadata: {},
};

const renderWith = (results: DailyData[]) => {
  mockUsePaginatedDailyActivity.mockReturnValue({ data: { results }, loading: false, isFetchingMore: false });
  mockUseQuery.mockImplementation((args: { queryKey: string[] }) =>
    args.queryKey[0] === "cost-optimization-usage-logs"
      ? {
          data: {
            logs: [
              {
                request_id: "req-123456789",
                timestamp: "2026-07-13T12:00:00Z",
                model: "test-model",
                total_tokens: 150,
                optimization_type: "both",
                spend: 0.02,
                savings: 0.14,
                original_cost: 0.16,
                compression_savings_spend: 0.1,
                prompt_caching_savings_spend: 0.04,
                tokens_saved: 100,
                cache_read_tokens: 50,
              },
            ],
            total: 1,
            page: 1,
            page_size: 50,
            total_pages: 1,
          },
          isLoading: false,
          isFetching: false,
          error: null,
        }
      : {
          data: { data: [detailLog], total: 1 },
          isLoading: false,
          isFetching: false,
          error: null,
        },
  );
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

  it("renders recent optimized requests with savings and optimization type", () => {
    const { getByText } = renderWith([day("2026-07-12", { compression_savings_spend: 0.04 })]);

    expect(getByText("Recent Optimized Requests")).toBeInTheDocument();
    expect(getByText("req-123456789")).toBeInTheDocument();
    expect(getByText("Both")).toBeInTheDocument();
    expect(getByText("$0.1600")).toBeInTheDocument();
    expect(getByText("$0.0200")).toBeInTheDocument();
    expect(getByText("$0.1400")).toBeInTheDocument();
  });

  it("fetches and opens request details when an optimized request is clicked", async () => {
    const { getByText, getByTestId } = renderWith([day("2026-07-12", { compression_savings_spend: 0.04 })]);

    fireEvent.click(getByText("req-123456789"));

    const detailQuery = mockUseQuery.mock.calls
      .map(([args]) => args as { queryKey: string[]; queryFn: () => Promise<unknown> })
      .find((args) => args.queryKey[0] === "cost-optimization-spend-log");
    expect(detailQuery).toBeDefined();
    await detailQuery?.queryFn();
    expect(mockUiSpendLogsCall).toHaveBeenCalledWith(
      expect.objectContaining({
        accessToken: "test-token",
        params: { request_id: "req-123456789" },
      }),
    );
    expect(getByTestId("log-details-drawer")).toHaveTextContent("req-123456789");
  });
});
