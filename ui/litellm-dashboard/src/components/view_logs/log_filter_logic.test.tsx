import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PaginatedResponse } from ".";
import type { LogEntry, LogsSortField } from "./columns";
import { useLogFilterLogic } from "./log_filter_logic";

vi.mock("../networking", () => ({
  uiSpendLogsCall: vi.fn(),
}));

vi.mock("@/components/key_team_helpers/filter_helpers", () => ({
  fetchAllKeyAliases: vi.fn().mockResolvedValue([]),
  fetchAllTeams: vi.fn().mockResolvedValue([]),
}));

import { uiSpendLogsCall } from "../networking";

const createLogEntry = (overrides: Partial<LogEntry> = {}): LogEntry =>
({
  request_id: "req-1",
  api_key: "key-1",
  team_id: "team-1",
  model: "gpt-4",
  model_id: "gpt-4",
  call_type: "chat",
  spend: 0,
  total_tokens: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  startTime: "2025-01-01T00:00:00Z",
  endTime: "2025-01-01T00:01:00Z",
  cache_hit: "miss",
  messages: [],
  response: {},
  metadata: {},
  request_tags: {},
  ...overrides,
} as LogEntry);

const createPaginatedResponse = (data: LogEntry[]): PaginatedResponse => ({
  data,
  total: data.length,
  page: 1,
  page_size: 50,
  total_pages: 1,
});

const defaultProps = {
  logs: createPaginatedResponse([]),
  accessToken: "test-token",
  startTime: "2025-01-01T00:00:00",
  endTime: "2025-01-01T23:59:59",
  isCustomDate: true,
  setCurrentPage: vi.fn(),
  userID: "user-1",
  userRole: "Admin",
};

describe("useLogFilterLogic", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    vi.clearAllMocks();
    vi.mocked(uiSpendLogsCall).mockResolvedValue({
      data: [],
      total: 0,
      page: 1,
      page_size: 50,
      total_pages: 0,
    });
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should return filters, filteredLogs, allKeyAliases, allTeams, handleFilterChange, and handleFilterReset", () => {
    const { result } = renderHook(
      () =>
        useLogFilterLogic({
          ...defaultProps,
          logs: createPaginatedResponse([createLogEntry()]),
        }),
      { wrapper },
    );

    expect(result.current.filters).toBeDefined();
    expect(result.current.filteredLogs).toBeDefined();
    expect(result.current.allKeyAliases).toBeDefined();
    expect(result.current).toHaveProperty("allTeams");
    expect(result.current.handleFilterChange).toBeDefined();
    expect(result.current.handleFilterReset).toBeDefined();
  });

  it("should initialize filters with all keys empty", () => {
    const { result } = renderHook(() => useLogFilterLogic(defaultProps), { wrapper });

    const filters = result.current.filters;
    expect(filters["Team ID"]).toBe("");
    expect(filters["Key Hash"]).toBe("");
    expect(filters["Request ID"]).toBe("");
    expect(filters["Model"]).toBe("");
    expect(filters["User ID"]).toBe("");
    expect(filters["End User"]).toBe("");
    expect(filters["Status"]).toBe("");
    expect(filters["Key Alias"]).toBe("");
    expect(filters["Error Code"]).toBe("");
    expect(filters["Error Message"]).toBe("");
  });

  it("should return all logs when no filters are applied", () => {
    const logs = createPaginatedResponse([
      createLogEntry({ request_id: "req-1" }),
      createLogEntry({ request_id: "req-2" }),
    ]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    expect(result.current.filteredLogs.data).toHaveLength(2);
    expect(result.current.filteredLogs.data).toEqual(logs.data);
  });

  it("should filter logs by team_id when Team ID filter is set", () => {
    const logs = createPaginatedResponse([
      createLogEntry({ request_id: "req-1", team_id: "team-a" }),
      createLogEntry({ request_id: "req-2", team_id: "team-b" }),
      createLogEntry({ request_id: "req-3", team_id: "team-a" }),
    ]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Team ID": "team-a" });
    });

    expect(result.current.filteredLogs.data).toHaveLength(2);
    expect(result.current.filteredLogs.data.every((log) => log.team_id === "team-a")).toBe(true);
  });

  it("should filter logs by status when Status filter is set to success", () => {
    const logs = createPaginatedResponse([
      createLogEntry({ request_id: "req-1", status: "success" }),
      createLogEntry({ request_id: "req-2" }),
      createLogEntry({ request_id: "req-3", status: "error" }),
    ]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ Status: "success" });
    });

    expect(result.current.filteredLogs.data).toHaveLength(2);
    expect(result.current.filteredLogs.data.every((log) => !log.status || log.status === "success")).toBe(true);
  });

  it("should filter logs by status when Status filter is set to error", () => {
    const logs = createPaginatedResponse([
      createLogEntry({ request_id: "req-1", status: "success" }),
      createLogEntry({ request_id: "req-2", status: "error" }),
    ]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ Status: "error" });
    });

    expect(result.current.filteredLogs.data).toHaveLength(1);
    expect(result.current.filteredLogs.data[0].status).toBe("error");
  });

  it("should filter logs by model_id when Model filter is set", async () => {
    const filteredLogs = [
      createLogEntry({ request_id: "req-1", model_id: "gpt-4" }),
      createLogEntry({ request_id: "req-3", model_id: "gpt-4" }),
    ];
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse(filteredLogs),
    );
    const logs = createPaginatedResponse([
      createLogEntry({ request_id: "req-1", model_id: "gpt-4" }),
      createLogEntry({ request_id: "req-2", model_id: "gpt-3.5" }),
      createLogEntry({ request_id: "req-3", model_id: "gpt-4" }),
    ]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ Model: "gpt-4" });
    });

    await waitFor(
      () => {
        expect(result.current.filteredLogs.data).toHaveLength(2);
        expect(result.current.filteredLogs.data.every((log) => log.model_id === "gpt-4")).toBe(true);
      },
      { timeout: 500 },
    );
  });

  it("should filter logs by api_key when Key Hash filter is set", async () => {
    const filteredLog = createLogEntry({ request_id: "req-1", api_key: "key-x" });
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([filteredLog]),
    );
    const logs = createPaginatedResponse([
      createLogEntry({ request_id: "req-1", api_key: "key-x" }),
      createLogEntry({ request_id: "req-2", api_key: "key-y" }),
    ]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Key Hash": "key-x" });
    });

    await waitFor(
      () => {
        expect(result.current.filteredLogs.data).toHaveLength(1);
        expect(result.current.filteredLogs.data[0].api_key).toBe("key-x");
      },
      { timeout: 500 },
    );
  });

  it("should filter logs by end_user when End User filter is set", async () => {
    const filteredLog = createLogEntry({ request_id: "req-1", end_user: "user-a" });
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([filteredLog]),
    );
    const logs = createPaginatedResponse([
      createLogEntry({ request_id: "req-1", end_user: "user-a" }),
      createLogEntry({ request_id: "req-2", end_user: "user-b" }),
    ]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "End User": "user-a" });
    });

    await waitFor(
      () => {
        expect(result.current.filteredLogs.data).toHaveLength(1);
        expect(result.current.filteredLogs.data[0].end_user).toBe("user-a");
      },
      { timeout: 500 },
    );
  });

  it("should filter logs by error_code when Error Code filter is set", async () => {
    const filteredLog = createLogEntry({
      request_id: "req-1",
      metadata: { error_information: { error_code: "429" } },
    });
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([filteredLog]),
    );
    const logs = createPaginatedResponse([
      createLogEntry({
        request_id: "req-1",
        metadata: { error_information: { error_code: "429" } },
      }),
      createLogEntry({
        request_id: "req-2",
        metadata: { error_information: { error_code: "500" } },
      }),
    ]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Error Code": "429" });
    });

    await waitFor(
      () => {
        expect(result.current.filteredLogs.data).toHaveLength(1);
        expect(result.current.filteredLogs.data[0].metadata?.error_information?.error_code).toBe("429");
      },
      { timeout: 500 },
    );
  });

  it("should return empty data when logs is null or has no data", () => {
    const { result } = renderHook(
      () =>
        useLogFilterLogic({
          ...defaultProps,
          logs: { data: [], total: 0, page: 1, page_size: 50, total_pages: 0 },
        }),
      { wrapper },
    );

    expect(result.current.filteredLogs.data).toEqual([]);
    expect(result.current.filteredLogs.total).toBe(0);
  });

  it("should reset filters when handleFilterReset is called", () => {
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Team ID": "team-1", Status: "success" });
    });

    expect(result.current.filters["Team ID"]).toBe("team-1");
    expect(result.current.filters["Status"]).toBe("success");

    act(() => {
      result.current.handleFilterReset();
    });

    expect(result.current.filters["Team ID"]).toBe("");
    expect(result.current.filters["Status"]).toBe("");
  });

  it("should call setCurrentPage with 1 when handleFilterChange is invoked", () => {
    const setCurrentPage = vi.fn();
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(
      () => useLogFilterLogic({ ...defaultProps, logs, setCurrentPage }),
      { wrapper },
    );

    act(() => {
      result.current.handleFilterChange({ "Team ID": "team-1" });
    });

    expect(setCurrentPage).toHaveBeenCalledWith(1);
  });

  it("should call uiSpendLogsCall when backend filter is set and debounce elapses", async () => {
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await waitFor(
      () => {
        expect(uiSpendLogsCall).toHaveBeenCalled();
      },
      { timeout: 500 },
    );
  });

  it("should not call uiSpendLogsCall when accessToken is null", async () => {
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(
      () => useLogFilterLogic({ ...defaultProps, logs, accessToken: null }),
      { wrapper },
    );

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await new Promise((resolve) => setTimeout(resolve, 350));

    expect(uiSpendLogsCall).not.toHaveBeenCalled();
  });

  it("should use backend filtered logs when backend filters are active and API returns data", async () => {
    const backendLog = createLogEntry({ request_id: "backend-req" });
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([backendLog]),
    );
    const logs = createPaginatedResponse([createLogEntry({ request_id: "client-req" })]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await waitFor(
      () => {
        expect(result.current.filteredLogs.data).toHaveLength(1);
        expect(result.current.filteredLogs.data[0].request_id).toBe("backend-req");
      },
      { timeout: 500 },
    );
  });

  it("should call uiSpendLogsCall with request_id when Request ID filter is set", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([createLogEntry({ request_id: "req-xyz" })]),
    );
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Request ID": "req-xyz" });
    });

    await waitFor(
      () => {
        expect(uiSpendLogsCall).toHaveBeenCalledWith(
          expect.objectContaining({
            params: expect.objectContaining({ request_id: "req-xyz" }),
          }),
        );
      },
      { timeout: 500 },
    );
  });

  it("should call uiSpendLogsCall with user_id when User ID filter is set", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([createLogEntry()]),
    );
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "User ID": "user-123" });
    });

    await waitFor(
      () => {
        expect(uiSpendLogsCall).toHaveBeenCalledWith(
          expect.objectContaining({
            params: expect.objectContaining({ user_id: "user-123" }),
          }),
        );
      },
      { timeout: 500 },
    );
  });

  it("should call uiSpendLogsCall with error_message when Error Message filter is set", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([createLogEntry()]),
    );
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Error Message": "rate limit exceeded" });
    });

    await waitFor(
      () => {
        expect(uiSpendLogsCall).toHaveBeenCalledWith(
          expect.objectContaining({
            params: expect.objectContaining({ error_message: "rate limit exceeded" }),
          }),
        );
      },
      { timeout: 500 },
    );
  });

  it("should fall back to logs when backend filters are active but API returns empty", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue({
      data: [],
      total: 0,
      page: 1,
      page_size: 50,
      total_pages: 0,
    });
    const clientLog = createLogEntry({ request_id: "client-req" });
    const logs = createPaginatedResponse([clientLog]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await waitFor(
      () => {
        expect(uiSpendLogsCall).toHaveBeenCalled();
      },
      { timeout: 500 },
    );

    expect(result.current.filteredLogs.data).toHaveLength(1);
    expect(result.current.filteredLogs.data[0].request_id).toBe("client-req");
  });

  it("should refetch when sortBy changes and backend filters are active", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([createLogEntry()]),
    );
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result, rerender } = renderHook(
      (props: { sortBy?: LogsSortField }) =>
        useLogFilterLogic({ ...defaultProps, logs, ...props }),
      { wrapper, initialProps: { sortBy: "startTime" as LogsSortField } },
    );

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(1), {
      timeout: 500,
    });

    rerender({ sortBy: "spend" as LogsSortField });

    await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(2), {
      timeout: 500,
    });
    expect(uiSpendLogsCall).toHaveBeenLastCalledWith(
      expect.objectContaining({
        params: expect.objectContaining({ sort_by: "spend" }),
      }),
    );
  });

  it("should refetch when sortOrder changes and backend filters are active", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([createLogEntry()]),
    );
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result, rerender } = renderHook(
      (props: { sortOrder?: "asc" | "desc" }) =>
        useLogFilterLogic({ ...defaultProps, logs, ...props }),
      { wrapper, initialProps: { sortOrder: "desc" } },
    );

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(1), {
      timeout: 500,
    });

    rerender({ sortOrder: "asc" });

    await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(2), {
      timeout: 500,
    });
    expect(uiSpendLogsCall).toHaveBeenLastCalledWith(
      expect.objectContaining({
        params: expect.objectContaining({ sort_order: "asc" }),
      }),
    );
  });

  it("should refetch when currentPage changes and backend filters are active", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([createLogEntry()]),
    );
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result, rerender } = renderHook(
      (props) => useLogFilterLogic({ ...defaultProps, logs, ...props }),
      { wrapper, initialProps: { currentPage: 1 } },
    );

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(1), {
      timeout: 500,
    });

    rerender({ currentPage: 2 });

    await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(2), {
      timeout: 500,
    });
    expect(uiSpendLogsCall).toHaveBeenLastCalledWith(
      expect.objectContaining({ page: 2 }),
    );
  });

  it("should not call setCurrentPage when handleFilterChange receives identical filters", async () => {
    const setCurrentPage = vi.fn();
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(
      () => useLogFilterLogic({ ...defaultProps, logs, setCurrentPage }),
      { wrapper },
    );

    act(() => {
      result.current.handleFilterChange({ "Team ID": "team-1" });
    });

    await waitFor(() => expect(setCurrentPage).toHaveBeenCalledTimes(1), {
      timeout: 500,
    });

    setCurrentPage.mockClear();

    await act(async () => {
      result.current.handleFilterChange({ "Team ID": "team-1" });
      await new Promise((resolve) => setTimeout(resolve, 350));
    });

    expect(setCurrentPage).not.toHaveBeenCalled();
  });

  it("should not crash when uiSpendLogsCall throws", async () => {
    vi.mocked(uiSpendLogsCall).mockRejectedValue(new Error("Network error"));
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled(), {
      timeout: 500,
    });

    expect(result.current.filteredLogs).toBeDefined();
    expect(result.current.filters).toBeDefined();
  });

  it("should clear backendFilteredLogs when handleFilterReset is called", async () => {
    const backendLog = createLogEntry({ request_id: "backend-req" });
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([backendLog]),
    );
    const logs = createPaginatedResponse([createLogEntry({ request_id: "client-req" })]);
    const { result } = renderHook(() => useLogFilterLogic({ ...defaultProps, logs }), { wrapper });

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await waitFor(
      () => {
        expect(result.current.filteredLogs.data[0].request_id).toBe("backend-req");
      },
      { timeout: 500 },
    );

    act(() => {
      result.current.handleFilterReset();
    });

    expect(result.current.filteredLogs.data).toEqual(logs.data);
    expect(result.current.filteredLogs.data[0].request_id).toBe("client-req");
  });

  it("should pass correct start_date, end_date, sort_by, and sort_order to uiSpendLogsCall", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue(
      createPaginatedResponse([createLogEntry()]),
    );
    const logs = createPaginatedResponse([createLogEntry()]);
    const { result } = renderHook(
      () =>
        useLogFilterLogic({
          ...defaultProps,
          logs,
          startTime: "2025-01-15T00:00:00Z",
          endTime: "2025-01-15T23:59:59Z",
          isCustomDate: true,
          sortBy: "spend",
          sortOrder: "asc",
        }),
      { wrapper },
    );

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "alias-1" });
    });

    await waitFor(
      () => {
        expect(uiSpendLogsCall).toHaveBeenCalledWith(
          expect.objectContaining({
            start_date: "2025-01-15 00:00:00",
            end_date: "2025-01-15 23:59:59",
            params: expect.objectContaining({
              sort_by: "spend",
              sort_order: "asc",
            }),
          }),
        );
      },
      { timeout: 500 },
    );
  });
});
