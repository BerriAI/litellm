import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SpendLogsTable from "./index";
import { renderWithProviders } from "../../../tests/test-utils";

const mockHandleFilterResetFromHook = vi.fn();
vi.mock("./log_filter_logic", () => ({
  useLogFilterLogic: vi.fn(() => ({
    logsQuery: { isLoading: false, isFetching: false, refetch: vi.fn() },
    filteredLogs: { data: [], total: 0, page: 1, page_size: 50, total_pages: 1 },
    allTeams: [],
    handleFilterChange: vi.fn(),
    handleFilterReset: mockHandleFilterResetFromHook,
  })),
  defaultFilters: {
    "Team ID": "",
    "Key Hash": "",
    "Request ID": "",
    "Model": "",
    "User ID": "",
    "End User": "",
    "Status": "",
    "Key Alias": "",
    "Error Code": "",
    "Error Message": "",
  },
}));

vi.mock("../networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../networking")>();
  return {
    ...actual,
    uiSpendLogsCall: vi.fn().mockResolvedValue({
      data: [],
      total: 0,
      page: 1,
      page_size: 50,
      total_pages: 0,
    }),
    keyListCall: vi.fn().mockResolvedValue({ keys: [] }),
    keyInfoV1Call: vi.fn().mockResolvedValue({ info: {} }),
    allEndUsersCall: vi.fn().mockResolvedValue([]),
  };
});

vi.mock("../key_team_helpers/filter_helpers", () => ({
  fetchAllTeams: vi.fn().mockResolvedValue([]),
}));

describe("SpendLogsTable", () => {
  const defaultProps = {
    accessToken: "test-token",
    token: "test-token",
    userRole: "Admin",
    userID: "user-1",
    premiumUser: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    // Clear sessionStorage to avoid isLiveTail state from previous tests
    sessionStorage.clear();
  });

  it("should call handleFilterResetFromHook when Reset Filters is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SpendLogsTable {...defaultProps} />);

    const resetButton = screen.getByRole("button", { name: "Reset Filters" });
    await user.click(resetButton);

    await waitFor(() => {
      expect(mockHandleFilterResetFromHook).toHaveBeenCalledTimes(1);
    });
  });

  it("should reset custom date range to default when Reset Filters is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SpendLogsTable {...defaultProps} />);

    // Open the time range quick select dropdown (button shows current range like "Last 24 Hours")
    const quickSelectButton = screen.getByRole("button", { name: /Last 24 Hours|Last 15 Minutes|Last Hour|Last 4 Hours|Last 7 Days/i });
    await user.click(quickSelectButton);

    // Click "Custom Range" to enable custom date selection
    const customRangeButton = await screen.findByRole("button", { name: "Custom Range" });
    await user.click(customRangeButton);

    // Custom date inputs should now be visible (start and end datetime-local inputs)
    const datetimeInputs = document.querySelectorAll('input[type="datetime-local"]');
    expect(datetimeInputs.length).toBeGreaterThanOrEqual(2);

    // Click Reset Filters - this should reset the custom date range and hide custom inputs
    const resetButton = screen.getByRole("button", { name: "Reset Filters" });
    await user.click(resetButton);

    await waitFor(() => {
      expect(mockHandleFilterResetFromHook).toHaveBeenCalled();
    });

    // After reset, custom date inputs should be hidden (isCustomDate reset to false)
    await waitFor(() => {
      const inputsAfterReset = document.querySelectorAll('input[type="datetime-local"]');
      expect(inputsAfterReset.length).toBe(0);
    });
  });
});
