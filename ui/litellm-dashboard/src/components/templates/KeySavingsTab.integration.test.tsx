import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import KeySavingsTab from "./KeySavingsTab";
import { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import * as useScopedDailyActivityRangeModule from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";

const metrics = (overrides: Partial<SpendMetrics>): SpendMetrics => ({
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

const day = (date: string, overrides: Partial<SpendMetrics>): DailyData => ({
  date,
  metrics: metrics(overrides),
  breakdown: {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
  },
});

const mockActivity = (
  overrides: Partial<useScopedDailyActivityRangeModule.DailyActivityRange> = {},
): useScopedDailyActivityRangeModule.DailyActivityRange => ({
  dateValue: { from: new Date("2025-01-01"), to: new Date("2025-01-31") },
  onDateChange: vi.fn(),
  results: [] as DailyData[],
  loading: false,
  isFetchingMore: false,
  progress: { currentPage: 1, totalPages: 1 },
  cancelled: false,
  cancel: vi.fn(),
  ...overrides,
});

const scopedRange = () => vi.spyOn(useScopedDailyActivityRangeModule, "useScopedDailyActivityRange");

const renderTab = (props: Partial<React.ComponentProps<typeof KeySavingsTab>> = {}) =>
  render(
    <KeySavingsTab
      accessToken="test-token"
      keyToken="key-abc123"
      userId="user-123"
      userRole="Internal User"
      {...props}
    />,
  );

describe("KeySavingsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("totals each savings driver across the days in range", () => {
    const firstDay: Partial<SpendMetrics> = {
      compression_savings_spend: 1.5,
      prompt_caching_savings_spend: 0.25,
      autorouter_savings_spend: 2,
      compression_saved_tokens: 400,
      cache_read_input_tokens: 300,
      prompt_tokens: 1000,
    };
    const secondDay: Partial<SpendMetrics> = {
      compression_savings_spend: 0.5,
      prompt_caching_savings_spend: 0.75,
      autorouter_savings_spend: 1,
      compression_saved_tokens: 600,
      cache_read_input_tokens: 200,
      prompt_tokens: 1000,
    };
    scopedRange().mockReturnValue(
      mockActivity({ results: [day("2025-01-01", firstDay), day("2025-01-02", secondDay)] }),
    );

    renderTab();

    expect(screen.getByTestId("summary-card-total-saved")).toHaveTextContent("$6.00");
    expect(screen.getByTestId("summary-card-compression-savings")).toHaveTextContent("$2.00");
    expect(screen.getByTestId("summary-card-compression-savings")).toHaveTextContent("1,000 tokens compressed");
    expect(screen.getByTestId("summary-card-prompt-caching-savings")).toHaveTextContent("$1.00");
    expect(screen.getByTestId("summary-card-auto-router-savings")).toHaveTextContent("$3.00");
  });

  it("separates a key with no traffic from one still loading", () => {
    scopedRange().mockReturnValue(mockActivity());

    const { unmount } = renderTab();
    expect(screen.getByTestId("key-savings-empty")).toHaveTextContent("No usage recorded for this key");
    unmount();

    scopedRange().mockReturnValue(mockActivity({ loading: true }));
    renderTab();
    expect(screen.getByTestId("key-savings-empty")).toHaveTextContent("Loading savings");
  });

  it("asks the endpoint for this key alone, scoped to the viewer's own rows", () => {
    const hook = scopedRange().mockReturnValue(mockActivity());

    renderTab({ userId: "user-456", userRole: "Internal User" });

    expect(hook).toHaveBeenCalledWith("test-token", { userId: "user-456", apiKey: "key-abc123" });
    expect(screen.getByTestId("key-savings-scope-note")).toHaveTextContent("Showing your own requests");
  });

  it("reads the whole key for a proxy admin, with no scope note to contradict it", () => {
    const hook = scopedRange().mockReturnValue(mockActivity());

    renderTab({ userId: "admin-123", userRole: "Admin" });

    expect(hook).toHaveBeenCalledWith("test-token", { userId: null, apiKey: "key-abc123" });
    expect(screen.queryByTestId("key-savings-scope-note")).not.toBeInTheDocument();
  });

  it("keeps the scope note for an org admin, whose figures cover only their own requests", () => {
    const hook = scopedRange().mockReturnValue(mockActivity());

    renderTab({ userId: "org-admin-1", userRole: "Org Admin" });

    expect(hook).toHaveBeenCalledWith("test-token", { userId: "org-admin-1", apiKey: "key-abc123" });
    expect(screen.getByTestId("key-savings-scope-note")).toHaveTextContent("Showing your own requests");
  });
});
