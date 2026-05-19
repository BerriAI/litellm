import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode, useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { LogsSortField } from "./columns";
import {
  defaultFilters,
  getLiveTailRefetchInterval,
  LIVE_TAIL_INTERVAL_MS,
  useLogFilterLogic,
  type LogFilterState,
  type PaginatedResponse,
} from "./log_filter_logic";

vi.mock("../networking", () => ({
  uiSpendLogsCall: vi.fn(),
}));

vi.mock("@/components/key_team_helpers/filter_helpers", () => ({
  fetchAllTeams: vi.fn().mockResolvedValue([]),
}));

import { uiSpendLogsCall } from "../networking";

const emptyResponse: PaginatedResponse = {
  data: [],
  total: 0,
  page: 1,
  page_size: 50,
  total_pages: 0,
};

const defaultProps = {
  accessToken: "test-token" as string | null,
  token: "test-token" as string | null,
  userRole: "Admin" as string | null,
  userID: "user-1" as string | null,
  filterByCurrentUser: false,
  activeTab: "request logs",
  isLiveTail: false,
  startTime: "2025-01-01T00:00:00",
  endTime: "2025-01-01T23:59:59",
  isCustomDate: true,
  sortBy: "startTime" as LogsSortField,
  sortOrder: "desc" as "asc" | "desc",
  currentPage: 1,
};

type HookOverrides = Partial<Omit<Parameters<typeof useLogFilterLogic>[0], "filters" | "setFilters">>;

describe("useLogFilterLogic", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    vi.clearAllMocks();
    vi.mocked(uiSpendLogsCall).mockResolvedValue(emptyResponse);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  function renderFilterHook(overrides: HookOverrides = {}) {
    const setCurrentPage = overrides.setCurrentPage ?? vi.fn();
    const rendered = renderHook(
      () => {
        const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
        const hook = useLogFilterLogic({
          ...defaultProps,
          ...overrides,
          filters,
          setFilters,
          setCurrentPage,
        });
        return { ...hook, filters, setFilters };
      },
      { wrapper },
    );
    return { ...rendered, setCurrentPage };
  }

  describe("return shape", () => {
    it("exposes filteredLogs, allTeams, handleFilterChange, handleFilterReset", () => {
      const { result } = renderFilterHook();

      expect(result.current.filteredLogs).toBeDefined();
      expect(result.current).toHaveProperty("allTeams");
      expect(result.current.handleFilterChange).toBeInstanceOf(Function);
      expect(result.current.handleFilterReset).toBeInstanceOf(Function);
    });
  });

  describe("handleFilterReset", () => {
    it("restores filters to defaults after changes", () => {
      const { result } = renderFilterHook();

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

    it("calls setCurrentPage(1)", () => {
      const setCurrentPage = vi.fn();
      const { result } = renderFilterHook({ setCurrentPage });

      act(() => {
        result.current.handleFilterReset();
      });

      expect(setCurrentPage).toHaveBeenCalledWith(1);
    });

    it("triggers a fetch with all filter params undefined", async () => {
      vi.mocked(uiSpendLogsCall).mockResolvedValue(emptyResponse);
      const { result } = renderFilterHook();

      act(() => {
        result.current.handleFilterChange({ "Key Alias": "alias-1" });
      });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled(), { timeout: 500 });

      act(() => {
        result.current.handleFilterReset();
      });

      await waitFor(
        () => {
          expect(uiSpendLogsCall).toHaveBeenLastCalledWith(
            expect.objectContaining({
              params: expect.objectContaining({
                team_id: undefined,
                api_key: undefined,
                request_id: undefined,
                user_id: undefined,
                end_user: undefined,
                status_filter: undefined,
                model_id: undefined,
                key_alias: undefined,
                error_code: undefined,
                error_message: undefined,
              }),
            }),
          );
        },
        { timeout: 500 },
      );
    });
  });

  describe("handleFilterChange", () => {
    it("calls setCurrentPage(1) when filters change", () => {
      const setCurrentPage = vi.fn();
      const { result } = renderFilterHook({ setCurrentPage });

      act(() => {
        result.current.handleFilterChange({ "Team ID": "team-1" });
      });

      expect(setCurrentPage).toHaveBeenCalledWith(1);
    });

    it("merges partial updates without clobbering other filter keys", () => {
      const { result } = renderFilterHook();

      act(() => {
        result.current.handleFilterChange({ "Team ID": "team-a" });
      });
      expect(result.current.filters["Team ID"]).toBe("team-a");

      act(() => {
        result.current.handleFilterChange({ Model: "gpt-4" });
      });

      expect(result.current.filters["Team ID"]).toBe("team-a");
      expect(result.current.filters["Model"]).toBe("gpt-4");
    });

    it("does not call setCurrentPage when filters are identical", async () => {
      const setCurrentPage = vi.fn();
      const { result } = renderFilterHook({ setCurrentPage });

      act(() => {
        result.current.handleFilterChange({ "Team ID": "team-1" });
      });

      await waitFor(() => expect(setCurrentPage).toHaveBeenCalledTimes(1), { timeout: 500 });

      setCurrentPage.mockClear();

      await act(async () => {
        result.current.handleFilterChange({ "Team ID": "team-1" });
        await new Promise((resolve) => setTimeout(resolve, 350));
      });

      expect(setCurrentPage).not.toHaveBeenCalled();
    });
  });

  describe("query params — filter keys", () => {
    const filterCases: Array<{
      filterKey: keyof LogFilterState;
      paramName: string;
      value: string;
    }> = [
      { filterKey: "Team ID", paramName: "team_id", value: "team-a" },
      { filterKey: "Key Hash", paramName: "api_key", value: "key-x" },
      { filterKey: "Request ID", paramName: "request_id", value: "req-xyz" },
      { filterKey: "User ID", paramName: "user_id", value: "user-123" },
      { filterKey: "End User", paramName: "end_user", value: "user-a" },
      { filterKey: "Status", paramName: "status_filter", value: "error" },
      { filterKey: "Model", paramName: "model_id", value: "gpt-4" },
      { filterKey: "Public model / search tool", paramName: "model", value: "tavily-marketing" },
      { filterKey: "Error Code", paramName: "error_code", value: "429" },
      { filterKey: "Error Message", paramName: "error_message", value: "rate limit exceeded" },
    ];

    it.each(filterCases)(
      "forwards $filterKey as params.$paramName to uiSpendLogsCall",
      async ({ filterKey, paramName, value }) => {
        const { result } = renderFilterHook();

        act(() => {
          result.current.handleFilterChange({ [filterKey]: value } as Partial<LogFilterState>);
        });

        await waitFor(
          () => {
            expect(uiSpendLogsCall).toHaveBeenCalledWith(
              expect.objectContaining({
                params: expect.objectContaining({ [paramName]: value }),
              }),
            );
          },
          { timeout: 500 },
        );
      },
    );
  });

  describe("query params — date & sort", () => {
    it("passes start_date, end_date, sort_by, and sort_order to uiSpendLogsCall", async () => {
      const { result } = renderFilterHook({
        startTime: "2025-01-15T00:00:00Z",
        endTime: "2025-01-15T23:59:59Z",
        isCustomDate: true,
        sortBy: "spend" as LogsSortField,
        sortOrder: "asc",
      });

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

  describe("debounce", () => {
    it("calls uiSpendLogsCall after the debounce elapses for text filters", async () => {
      const { result } = renderFilterHook();

      act(() => {
        result.current.handleFilterChange({ "Key Hash": "hash-1" });
      });

      await waitFor(
        () =>
          expect(uiSpendLogsCall).toHaveBeenCalledWith(
            expect.objectContaining({
              params: expect.objectContaining({ api_key: "hash-1" }),
            }),
          ),
        { timeout: 500 },
      );
    });

    it("does not call uiSpendLogsCall with a text filter before the debounce elapses", async () => {
      const { result } = renderFilterHook();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled(), { timeout: 500 });
      vi.mocked(uiSpendLogsCall).mockClear();

      act(() => {
        result.current.handleFilterChange({ "Key Hash": "hash-1" });
      });

      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(uiSpendLogsCall).not.toHaveBeenCalledWith(
        expect.objectContaining({
          params: expect.objectContaining({ api_key: "hash-1" }),
        }),
      );

      await waitFor(
        () =>
          expect(uiSpendLogsCall).toHaveBeenCalledWith(
            expect.objectContaining({
              params: expect.objectContaining({ api_key: "hash-1" }),
            }),
          ),
        { timeout: 500 },
      );
    });

    it("applies dropdown filter changes without waiting for the debounce", async () => {
      const { result } = renderFilterHook();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled(), { timeout: 500 });
      vi.mocked(uiSpendLogsCall).mockClear();

      act(() => {
        result.current.handleFilterChange({ "Team ID": "team-instant" });
      });

      await waitFor(
        () =>
          expect(uiSpendLogsCall).toHaveBeenCalledWith(
            expect.objectContaining({
              params: expect.objectContaining({ team_id: "team-instant" }),
            }),
          ),
        { timeout: 100 },
      );
    });

    // Guards the TEXT_FILTER_KEYS fix: this free-text filter must debounce, not fire per keystroke.
    it("debounces the 'Public model / search tool' text filter", async () => {
      const { result } = renderFilterHook();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled(), { timeout: 500 });
      vi.mocked(uiSpendLogsCall).mockClear();

      act(() => {
        result.current.handleFilterChange({ "Public model / search tool": "tavily-marketing" });
      });

      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(uiSpendLogsCall).not.toHaveBeenCalledWith(
        expect.objectContaining({
          params: expect.objectContaining({ model: "tavily-marketing" }),
        }),
      );

      await waitFor(
        () =>
          expect(uiSpendLogsCall).toHaveBeenCalledWith(
            expect.objectContaining({
              params: expect.objectContaining({ model: "tavily-marketing" }),
            }),
          ),
        { timeout: 500 },
      );
    });
  });

  describe("handleFilterReset", () => {
    it("flushes the text-filter debounce so a pending typed value is not sent", async () => {
      const { result } = renderFilterHook();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled(), { timeout: 500 });
      vi.mocked(uiSpendLogsCall).mockClear();

      act(() => {
        result.current.handleFilterChange({ "Key Hash": "pending-hash" });
      });

      act(() => {
        result.current.handleFilterReset();
      });

      await new Promise((resolve) => setTimeout(resolve, 400));

      for (const call of vi.mocked(uiSpendLogsCall).mock.calls) {
        expect(call[0].params?.api_key).toBeUndefined();
      }
    });
  });

  describe("backend filtered logs", () => {
    it("returns the query payload as filteredLogs when backend filters are active", async () => {
      const backendLog = { request_id: "backend-req" };
      vi.mocked(uiSpendLogsCall).mockResolvedValue({
        data: [backendLog],
        total: 1,
        page: 1,
        page_size: 50,
        total_pages: 1,
      } as PaginatedResponse);

      const { result } = renderFilterHook();

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

    it("returns empty data when the API returns an empty payload", async () => {
      vi.mocked(uiSpendLogsCall).mockResolvedValue(emptyResponse);
      const { result } = renderFilterHook();

      act(() => {
        result.current.handleFilterChange({ "Key Alias": "alias-1" });
      });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled(), { timeout: 500 });

      expect(result.current.filteredLogs.data).toHaveLength(0);
    });
  });

  describe("refetch triggers", () => {
    it("refetches when sortBy changes", async () => {
      const { rerender } = renderHook(
        (props: { sortBy: LogsSortField }) => {
          const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
          return useLogFilterLogic({
            ...defaultProps,
            filters,
            setFilters,
            setCurrentPage: vi.fn(),
            sortBy: props.sortBy,
          });
        },
        { wrapper, initialProps: { sortBy: "startTime" } },
      );

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(1), { timeout: 500 });

      rerender({ sortBy: "spend" });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(2), { timeout: 500 });
      expect(uiSpendLogsCall).toHaveBeenLastCalledWith(
        expect.objectContaining({
          params: expect.objectContaining({ sort_by: "spend" }),
        }),
      );
    });

    it("refetches when sortOrder changes", async () => {
      const { rerender } = renderHook(
        (props: { sortOrder: "asc" | "desc" }) => {
          const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
          return useLogFilterLogic({
            ...defaultProps,
            filters,
            setFilters,
            setCurrentPage: vi.fn(),
            sortOrder: props.sortOrder,
          });
        },
        { wrapper, initialProps: { sortOrder: "desc" } },
      );

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(1), { timeout: 500 });

      rerender({ sortOrder: "asc" });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(2), { timeout: 500 });
      expect(uiSpendLogsCall).toHaveBeenLastCalledWith(
        expect.objectContaining({
          params: expect.objectContaining({ sort_order: "asc" }),
        }),
      );
    });

    it("refetches when currentPage changes", async () => {
      const { rerender } = renderHook(
        (props: { currentPage: number }) => {
          const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
          return useLogFilterLogic({
            ...defaultProps,
            filters,
            setFilters,
            setCurrentPage: vi.fn(),
            currentPage: props.currentPage,
          });
        },
        { wrapper, initialProps: { currentPage: 1 } },
      );

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(1), { timeout: 500 });

      rerender({ currentPage: 2 });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(2), { timeout: 500 });
      expect(uiSpendLogsCall).toHaveBeenLastCalledWith(expect.objectContaining({ page: 2 }));
    });

    it("refetches when startTime changes", async () => {
      const { rerender } = renderHook(
        (props: { startTime: string }) => {
          const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
          return useLogFilterLogic({
            ...defaultProps,
            filters,
            setFilters,
            setCurrentPage: vi.fn(),
            startTime: props.startTime,
          });
        },
        { wrapper, initialProps: { startTime: "2025-01-01T00:00:00Z" } },
      );

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(1), { timeout: 500 });

      rerender({ startTime: "2025-01-02T00:00:00Z" });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(2), { timeout: 500 });
      expect(uiSpendLogsCall).toHaveBeenLastCalledWith(expect.objectContaining({ start_date: "2025-01-02 00:00:00" }));
    });

    it("refetches with a different end_date when isCustomDate toggles", async () => {
      const customEndTime = "2025-01-15T23:59:59Z";
      const customEndFormatted = "2025-01-15 23:59:59";

      const { rerender } = renderHook(
        (props: { isCustomDate: boolean }) => {
          const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
          return useLogFilterLogic({
            ...defaultProps,
            endTime: customEndTime,
            filters,
            setFilters,
            setCurrentPage: vi.fn(),
            isCustomDate: props.isCustomDate,
          });
        },
        { wrapper, initialProps: { isCustomDate: false } },
      );

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(1), { timeout: 500 });
      const firstEndDate = vi.mocked(uiSpendLogsCall).mock.calls[0][0].end_date;
      expect(firstEndDate).not.toBe(customEndFormatted);

      rerender({ isCustomDate: true });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(2), { timeout: 500 });
      expect(vi.mocked(uiSpendLogsCall).mock.calls[1][0].end_date).toBe(customEndFormatted);
    });
  });

  describe("query enablement", () => {
    const nullCredentialCases: Array<{ name: string; override: HookOverrides }> = [
      { name: "accessToken", override: { accessToken: null } },
      { name: "token", override: { token: null } },
      { name: "userRole", override: { userRole: null } },
      { name: "userID", override: { userID: null } },
    ];

    it.each(nullCredentialCases)("does not call uiSpendLogsCall when $name is null", async ({ override }) => {
      const { result } = renderFilterHook(override);

      act(() => {
        result.current.handleFilterChange({ "Key Alias": "alias-1" });
      });

      await new Promise((resolve) => setTimeout(resolve, 350));

      expect(uiSpendLogsCall).not.toHaveBeenCalled();
    });

    it("does not call uiSpendLogsCall when activeTab is not 'request logs'", async () => {
      const { result } = renderFilterHook({ activeTab: "audit logs" });

      act(() => {
        result.current.handleFilterChange({ "Key Alias": "alias-1" });
      });

      await new Promise((resolve) => setTimeout(resolve, 350));

      expect(uiSpendLogsCall).not.toHaveBeenCalled();
    });
  });

  describe("filterByCurrentUser", () => {
    it("sends user_id: userID when the User ID filter is blank", async () => {
      const { result } = renderFilterHook({
        filterByCurrentUser: true,
        userID: "me-123",
      });

      act(() => {
        result.current.handleFilterChange({ "Key Alias": "alias-1" });
      });

      await waitFor(
        () => {
          expect(uiSpendLogsCall).toHaveBeenCalledWith(
            expect.objectContaining({
              params: expect.objectContaining({ user_id: "me-123" }),
            }),
          );
        },
        { timeout: 500 },
      );
    });
  });

  describe("error handling", () => {
    it("does not crash when uiSpendLogsCall throws", async () => {
      vi.mocked(uiSpendLogsCall).mockRejectedValue(new Error("Network error"));
      const { result } = renderFilterHook();

      act(() => {
        result.current.handleFilterChange({ "Key Alias": "alias-1" });
      });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled(), { timeout: 500 });

      expect(result.current.filteredLogs).toBeDefined();
      expect(result.current.filteredLogs.data).toEqual([]);
    });
  });
});

describe("getLiveTailRefetchInterval", () => {
  it("polls every 15s when live tail is on and on page 1", () => {
    expect(getLiveTailRefetchInterval(true, 1)).toBe(LIVE_TAIL_INTERVAL_MS);
  });

  it("does not poll when live tail is off", () => {
    expect(getLiveTailRefetchInterval(false, 1)).toBe(false);
  });

  it("does not poll when not on page 1, even with live tail on", () => {
    expect(getLiveTailRefetchInterval(true, 2)).toBe(false);
  });
});
