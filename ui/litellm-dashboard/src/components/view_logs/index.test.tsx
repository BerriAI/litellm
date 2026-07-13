import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import moment from "moment";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SpendLogsTable from "./index";
import { renderWithProviders } from "../../../tests/test-utils";
import { uiSpendLogsCall } from "../networking";
import { useLogFilterLogic } from "./log_filter_logic";

const mockHandleFilterResetFromHook = vi.fn();
vi.mock("./log_filter_logic", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./log_filter_logic")>();
  return {
    ...actual,
    useLogFilterLogic: vi.fn(() => ({
      logsQuery: { isLoading: false, isFetching: false, refetch: vi.fn() },
      filteredLogs: { data: [], total: 0, page: 1, page_size: 50, total_pages: 1 },
      allTeams: [],
      handleFilterChange: vi.fn(),
      handleFilterReset: mockHandleFilterResetFromHook,
    })),
  };
});

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
    const quickSelectButton = screen.getByRole("button", {
      name: /Last 24 Hours|Last 15 Minutes|Last Hour|Last 4 Hours|Last 7 Days/i,
    });
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

  describe("auth-not-ready guard", () => {
    it("shows a loading spinner when credentials are not yet resolved", () => {
      renderWithProviders(<SpendLogsTable {...defaultProps} accessToken={null} />);

      expect(document.querySelector(".ant-spin")).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Reset Filters" })).not.toBeInTheDocument();
    });

    it("renders the table (no spinner) once all credentials are present", () => {
      renderWithProviders(<SpendLogsTable {...defaultProps} />);

      expect(document.querySelector(".ant-spin")).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Reset Filters" })).toBeInTheDocument();
    });
  });

  describe("session rows are not collapsed", () => {
    const makeLog = (overrides: Partial<{
      request_id: string;
      session_id: string;
      call_type: string;
      session_total_count: number;
      model: string;
    }>) => ({
      request_id: overrides.request_id ?? "req-1",
      api_key: "key-hash",
      team_id: "team-1",
      model: overrides.model ?? "gpt-4o",
      model_id: "model-1",
      call_type: overrides.call_type ?? "acompletion",
      spend: 0.01,
      total_tokens: 10,
      prompt_tokens: 5,
      completion_tokens: 5,
      startTime: "2026-07-13 20:00:00",
      endTime: "2026-07-13 20:00:01",
      user: "user-1",
      cache_hit: "None",
      messages: [],
      response: {},
      session_id: overrides.session_id,
      session_total_count: overrides.session_total_count,
      status: "success",
    });

    beforeEach(() => {
      vi.mocked(useLogFilterLogic).mockReturnValue({
        logsQuery: {
          isLoading: false,
          isFetching: false,
          isPlaceholderData: false,
          refetch: vi.fn(),
        },
        filteredLogs: {
          data: [
            makeLog({
              request_id: "req-session-a-1",
              session_id: "session-a",
              session_total_count: 3,
              call_type: "acompletion",
            }),
            makeLog({
              request_id: "req-session-a-2",
              session_id: "session-a",
              session_total_count: 3,
              call_type: "call_mcp_tool",
            }),
            makeLog({
              request_id: "req-session-a-3",
              session_id: "session-a",
              session_total_count: 3,
              call_type: "acompletion",
            }),
            makeLog({
              request_id: "req-session-b-1",
              session_id: "session-b",
              session_total_count: 2,
              call_type: "acompletion",
            }),
            makeLog({
              request_id: "req-session-b-2",
              session_id: "session-b",
              session_total_count: 2,
              call_type: "acompletion",
            }),
          ],
          total: 5,
          page: 1,
          page_size: 50,
          total_pages: 1,
        },
        allTeams: [],
        handleFilterChange: vi.fn(),
        handleFilterReset: mockHandleFilterResetFromHook,
      } as unknown as ReturnType<typeof useLogFilterLogic>);
    });

    it("renders every spend-log row even when multiple share a session_id", async () => {
      renderWithProviders(<SpendLogsTable {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("req-session-a-1")).toBeInTheDocument();
      });
      expect(screen.getByText("req-session-a-2")).toBeInTheDocument();
      expect(screen.getByText("req-session-a-3")).toBeInTheDocument();
      expect(screen.getByText("req-session-b-1")).toBeInTheDocument();
      expect(screen.getByText("req-session-b-2")).toBeInTheDocument();
      expect(screen.getByText(/Showing 1 - 5 of 5 results/)).toBeInTheDocument();
    });
  });

  describe("Quick Select time range", () => {
    // uiSpendLogsCall fires from the real useLogFilterLogic query, so restore it here.
    beforeEach(async () => {
      const actual = await vi.importActual<typeof import("./log_filter_logic")>("./log_filter_logic");
      vi.mocked(useLogFilterLogic).mockImplementation(actual.useLogFilterLogic);
    });

    const waitForWindowSeconds = async (minMinutes: number) => {
      let diff = -1;
      await waitFor(() => {
        const lastCall = vi.mocked(uiSpendLogsCall).mock.calls.at(-1)?.[0];
        if (!lastCall) throw new Error("uiSpendLogsCall was not called");
        diff = moment
          .utc(lastCall.end_date, "YYYY-MM-DD HH:mm:ss")
          .diff(moment.utc(lastCall.start_date, "YYYY-MM-DD HH:mm:ss"), "seconds");
        // start_date is rounded down to the minute boundary, end_date is the
        // current wall-clock at queryFn time. The dropped sub-minute fraction
        // on start_date can push the diff up to (minMinutes+1)*60 seconds
        // exactly (e.g. click at HH:MM:59.9 → start floors to HH:MM:00 and
        // queryFn fires just past HH:(MM+1):00), so allow equality on the
        // upper bound.
        expect(diff).toBeGreaterThanOrEqual(minMinutes * 60);
        expect(diff).toBeLessThanOrEqual((minMinutes + 1) * 60);
      });
      return diff;
    };

    it("should pass a ~1-minute window to uiSpendLogsCall when 'Last Minute' is selected", async () => {
      const user = userEvent.setup();
      renderWithProviders(<SpendLogsTable {...defaultProps} />);

      await user.click(screen.getByRole("button", { name: /Last 24 Hours/i }));
      await user.click(await screen.findByRole("button", { name: "Last Minute" }));

      await waitForWindowSeconds(1);
    });

    it("should pass a ~15-minute window to uiSpendLogsCall when 'Last 15 Minutes' is selected", async () => {
      const user = userEvent.setup();
      renderWithProviders(<SpendLogsTable {...defaultProps} />);

      await user.click(screen.getByRole("button", { name: /Last 24 Hours/i }));
      await user.click(await screen.findByRole("button", { name: "Last 15 Minutes" }));

      await waitForWindowSeconds(15);
    });

    it("should update the time-range button label to 'Last Minute' after selecting it", async () => {
      const user = userEvent.setup();
      renderWithProviders(<SpendLogsTable {...defaultProps} />);

      await user.click(screen.getByRole("button", { name: /Last 24 Hours/i }));
      await user.click(await screen.findByRole("button", { name: "Last Minute" }));

      expect(screen.getByRole("button", { name: "Last Minute" })).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /Last 24 Hours/i })).not.toBeInTheDocument();
    });
  });
});
