import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SpendLogsTable, { RequestViewer } from "./index";
import type { LogEntry } from "./columns";
import type { Row } from "@tanstack/react-table";
import type { Team } from "../key_team_helpers/key_list";
import { renderWithProviders } from "../../../tests/test-utils";

const mockHandleFilterResetFromHook = vi.fn();
vi.mock("./log_filter_logic", () => ({
  useLogFilterLogic: vi.fn(() => ({
    filters: {},
    filteredLogs: { data: [], total: 0, page: 1, page_size: 50, total_pages: 1 },
    allTeams: [],
    allKeyAliases: [],
    handleFilterChange: vi.fn(),
    handleFilterReset: mockHandleFilterResetFromHook,
  })),
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
  fetchAllKeyAliases: vi.fn().mockResolvedValue([]),
  fetchAllTeams: vi.fn().mockResolvedValue([]),
}));

const baseLogEntry: LogEntry = {
  request_id: "chatcmpl-test-id",
  api_key: "api-key",
  team_id: "team-id",
  model: "gpt-4",
  model_id: "gpt-4",
  call_type: "chat",
  spend: 0,
  total_tokens: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  startTime: "2025-11-14T00:00:00Z",
  endTime: "2025-11-14T00:00:00Z",
  cache_hit: "miss",
  duration: 1,
  messages: [{ role: "user", content: "hello" }],
  response: { status: "ok" },
  metadata: {
    status: "success",
    additional_usage_values: {
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
  },
  request_tags: {},
  custom_llm_provider: "openai",
  api_base: "https://api.example.com",
};

const createRow = (overrides: Partial<LogEntry> = {}): Row<LogEntry> =>
  ({
    original: {
      ...baseLogEntry,
      ...overrides,
    },
  }) as unknown as Row<LogEntry>;

describe("Request Viewer", () => {
  it("renders the request details heading", () => {
    render(<RequestViewer row={createRow()} />);
    expect(screen.getByText("Request Details")).toBeInTheDocument();
  });

  it("should truncate the request id if it is longer than 64 characters", () => {
    const LONG_REQUEST_ID = "a".repeat(128);
    const TRUNCATED_REQUEST_ID = `${"a".repeat(64)}...`;
    render(
      <RequestViewer
        row={createRow({
          request_id: LONG_REQUEST_ID,
        })}
      />,
    );

    expect(screen.getByText(TRUNCATED_REQUEST_ID)).toBeInTheDocument();
  });

  it("should display LiteLLM Overhead when litellm_overhead_time_ms is present in metadata", () => {
    render(
      <RequestViewer
        row={createRow({
          metadata: {
            status: "success",
            litellm_overhead_time_ms: 150,
            additional_usage_values: {
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
          },
        })}
      />,
    );

    expect(screen.getByText("LiteLLM Overhead:")).toBeInTheDocument();
    expect(screen.getByText("150 ms")).toBeInTheDocument();
  });

  it("should not display LiteLLM Overhead when litellm_overhead_time_ms is not present in metadata", () => {
    render(<RequestViewer row={createRow()} />);

    expect(screen.queryByText("LiteLLM Overhead:")).not.toBeInTheDocument();
  });
});

describe("SpendLogsTable", () => {
  const defaultProps = {
    accessToken: "test-token",
    token: "test-token",
    userRole: "Admin",
    userID: "user-1",
    allTeams: [] as Team[],
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
