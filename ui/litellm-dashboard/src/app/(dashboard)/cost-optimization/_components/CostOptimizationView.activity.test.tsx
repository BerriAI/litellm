import { fireEvent, render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mockUserDailyActivityCall = vi.fn();

vi.mock("@/components/networking", () => ({
  userDailyActivityCall: (...args: unknown[]) => mockUserDailyActivityCall(...args),
  getToolSpend: vi.fn().mockResolvedValue({ by_tool: [], daily: [], total_spend: 0, start_date: null, end_date: null }),
  getGeneralSettingsCall: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/shared/advanced_date_picker", () => ({
  __esModule: true,
  default: () => <div data-testid="date-picker" />,
}));

vi.mock("@/components/shared/charts", () => ({
  AreaChart: () => <div />,
  DonutChart: () => <div />,
  BarChart: () => <div />,
  DEFAULT_COLOR_CYCLE: ["emerald"],
}));

vi.mock("@/app/(dashboard)/router-settings/_components/general_settings", () => ({
  PromptCachingPanel: () => <div data-testid="caching-settings" />,
}));

vi.mock("./PromptCompressionTab", () => ({ __esModule: true, default: () => <div /> }));
vi.mock("./AutorouterTab", () => ({ __esModule: true, default: () => <div /> }));

import CostOptimizationView from "./CostOptimizationView";

const singlePage = {
  results: [],
  metadata: { total_pages: 1, has_more: false, page: 1 },
};

describe("CostOptimizationView daily activity", () => {
  it("fetches daily activity once for the page and shares it with every tab that needs it", async () => {
    mockUserDailyActivityCall.mockResolvedValue(singlePage);

    const { getByRole, getByTestId } = render(
      <CostOptimizationView accessToken="test-token" userId="u1" userRole="proxy_admin" />,
    );

    await waitFor(() => expect(mockUserDailyActivityCall).toHaveBeenCalledTimes(1));

    fireEvent.click(getByRole("tab", { name: "Prompt Caching" }));
    await waitFor(() => expect(getByTestId("caching-settings")).toBeInTheDocument());

    expect(mockUserDailyActivityCall).toHaveBeenCalledTimes(1);
  });
});
