import React from "react";
import { fireEvent, render, waitFor, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const mockUserDailyActivityCall = vi.fn();
const mockUserDailyActivityAggregatedCall = vi.fn();
const { useAuthorizedMock, mockToolSpendResponse } = vi.hoisted(() => ({
  useAuthorizedMock: vi.fn(),
  mockToolSpendResponse: { by_tool: [], daily: [], start_date: null, end_date: null },
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: useAuthorizedMock,
}));

vi.mock("@/components/networking", () => ({
  userDailyActivityCall: (...args: unknown[]) => mockUserDailyActivityCall(...args),
  userDailyActivityAggregatedCall: (...args: unknown[]) => mockUserDailyActivityAggregatedCall(...args),
  getToolSpend: vi.fn().mockResolvedValue(mockToolSpendResponse),
  getGeneralSettingsCall: vi.fn().mockResolvedValue([]),
  organizationListCall: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/shared/advanced_date_picker", () => ({
  __esModule: true,
  default: () => <div data-testid="date-picker" />,
}));

vi.mock("@/components/shared/charts", () => ({
  AreaChart: () => <div />,
  DonutChart: () => <div />,
  BarChart: () => <div />,
  CustomLegend: () => <div />,
  chartColorValue: (color: string) => color,
  DEFAULT_COLOR_CYCLE: ["blue", "cyan", "sky", "indigo", "violet", "purple", "fuchsia", "slate"],
  SEQUENTIAL_COLOR_RAMP: ["indigo"],
}));

vi.mock("@/app/(dashboard)/router-settings/_components/general_settings", () => ({
  PromptCachingPanel: () => <div data-testid="caching-settings" />,
}));

vi.mock("./PromptCompressionTab", () => ({ __esModule: true, default: () => <div /> }));

import CostOptimizationView from "./CostOptimizationView";

const singlePage = {
  results: [],
  metadata: { total_pages: 1, has_more: false, page: 1 },
};

describe("CostOptimizationView daily activity", () => {
  it("fetches daily activity once for the page and shares it with every tab that needs it", async () => {
    mockUserDailyActivityAggregatedCall.mockResolvedValue(singlePage);
    useAuthorizedMock.mockReturnValue({ accessToken: "test-token", userId: "u1", userRole: "proxy_admin" });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    render(
      <QueryClientProvider client={queryClient}>
        <CostOptimizationView accessToken="test-token" userId="u1" userRole="proxy_admin" />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("tab", { name: "Prompt Caching" }));
    await screen.findByTestId("caching-settings");

    expect(mockUserDailyActivityAggregatedCall).toHaveBeenCalledTimes(1);
    expect(mockUserDailyActivityCall).not.toHaveBeenCalled();
    expect(screen.queryByText(/Currently fetching spend data/)).not.toBeInTheDocument();
  });

  it("shows the fetch-progress banner while the paginated fallback streams pages in", async () => {
    mockUserDailyActivityAggregatedCall.mockReset();
    mockUserDailyActivityCall.mockReset();
    mockUserDailyActivityAggregatedCall.mockRejectedValue(new Error("aggregated unavailable"));
    mockUserDailyActivityCall.mockImplementation((...args: unknown[]) =>
      args[3] === 1
        ? Promise.resolve({ results: [], metadata: { total_pages: 3, has_more: true, page: 1 } })
        : new Promise(() => {}),
    );
    useAuthorizedMock.mockReturnValue({ accessToken: "test-token", userId: "u1", userRole: "proxy_admin" });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    render(
      <QueryClientProvider client={queryClient}>
        <CostOptimizationView accessToken="test-token" userId="u1" userRole="proxy_admin" />
      </QueryClientProvider>,
    );

    expect(await screen.findByText(/Currently fetching spend data: fetched 1 \/ 3 pages/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
  });
});
