import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ColumnFiltersState, PaginationState, SortingState } from "@tanstack/react-table";
import { renderHook, waitFor } from "@testing-library/react";
import moment from "moment";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_LOGS_SORTING,
  getFilterValue,
  getLiveTailRefetchInterval,
  LIVE_TAIL_INTERVAL_MS,
  LOG_FILTER_IDS,
  useLogFilterLogic,
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

const FIRST_PAGE: PaginationState = { pageIndex: 0, pageSize: 50 };

const defaultProps = {
  accessToken: "test-token" as string | null,
  token: "test-token" as string | null,
  userRole: "Admin" as string | null,
  userID: "user-1" as string | null,
  columnFilters: [] as ColumnFiltersState,
  filterByCurrentUser: false,
  activeTab: "request logs",
  isLiveTail: false,
  startTime: "2025-01-01T00:00:00",
  endTime: "2025-01-01T23:59:59",
  pagination: FIRST_PAGE,
  isCustomDate: true,
  sorting: DEFAULT_LOGS_SORTING,
};

type HookOverrides = Partial<Parameters<typeof useLogFilterLogic>[0]>;

const lastCallParams = () => vi.mocked(uiSpendLogsCall).mock.calls.at(-1)?.[0];

describe("useLogFilterLogic", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
    vi.mocked(uiSpendLogsCall).mockResolvedValue(emptyResponse);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  function renderFilterHook(overrides: HookOverrides = {}) {
    return renderHook(() => useLogFilterLogic({ ...defaultProps, ...overrides }), { wrapper });
  }

  describe("column filters map onto backend query params", () => {
    const cases: ReadonlyArray<{ id: string; value: string; param: string }> = [
      { id: LOG_FILTER_IDS.KEY_HASH, value: "sk-hash-1", param: "api_key" },
      { id: LOG_FILTER_IDS.TEAM_ID, value: "team-1", param: "team_id" },
      { id: LOG_FILTER_IDS.REQUEST_ID, value: "req-1", param: "request_id" },
      { id: LOG_FILTER_IDS.SESSION_ID, value: "sess-1", param: "session_id" },
      { id: LOG_FILTER_IDS.END_USER, value: "end-user-1", param: "end_user" },
      { id: LOG_FILTER_IDS.STATUS, value: "failure", param: "status_filter" },
      { id: LOG_FILTER_IDS.MODEL_ID, value: "model-uuid-1", param: "model_id" },
      { id: LOG_FILTER_IDS.PUBLIC_MODEL_OR_SEARCH_TOOL, value: "gpt-4o", param: "model" },
      { id: LOG_FILTER_IDS.KEY_ALIAS, value: "alias-1", param: "key_alias" },
      { id: LOG_FILTER_IDS.ERROR_CODE, value: "429", param: "error_code" },
      { id: LOG_FILTER_IDS.ERROR_MESSAGE, value: "rate limited", param: "error_message" },
      { id: LOG_FILTER_IDS.USER_ID, value: "user-9", param: "user_id" },
    ];

    it.each(cases)("sends $id as $param", async ({ id, value, param }) => {
      renderFilterHook({ columnFilters: [{ id, value }] });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()?.params).toMatchObject({ [param]: value });
    });

    it("omits params for filters that are absent, blank, or whitespace-only", async () => {
      renderFilterHook({
        columnFilters: [
          { id: LOG_FILTER_IDS.TEAM_ID, value: "   " },
          { id: LOG_FILTER_IDS.KEY_HASH, value: "" },
        ],
      });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      const params = lastCallParams()?.params;
      expect(params?.team_id).toBeUndefined();
      expect(params?.api_key).toBeUndefined();
      expect(params?.error_code).toBeUndefined();
    });
  });

  describe("paging, dates, and sort", () => {
    it("sends a 1-based page derived from pageIndex", async () => {
      renderFilterHook({ pagination: { pageIndex: 2, pageSize: 25 } });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()).toMatchObject({ page: 3, page_size: 25 });
    });

    it("passes start_date, end_date, sort_by, and sort_order", async () => {
      renderFilterHook({ sorting: [{ id: "spend", desc: false }] });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      const call = lastCallParams();
      expect(call?.start_date).toBe(moment(defaultProps.startTime).utc().format("YYYY-MM-DD HH:mm:ss"));
      expect(call?.end_date).toBe(moment(defaultProps.endTime).utc().format("YYYY-MM-DD HH:mm:ss"));
      expect(call?.params).toMatchObject({ sort_by: "spend", sort_order: "asc" });
    });

    it("falls back to the default sort when the sorting state is empty", async () => {
      renderFilterHook({ sorting: [] });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()?.params).toMatchObject({ sort_by: "startTime", sort_order: "desc" });
    });

    it("ignores a sort id the backend does not support", async () => {
      renderFilterHook({ sorting: [{ id: "request_id", desc: false }] as SortingState });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()?.params).toMatchObject({ sort_by: "startTime" });
    });
  });

  describe("refetch triggers", () => {
    it.each([
      ["sorting", { sorting: [{ id: "spend", desc: true }] as SortingState }],
      ["pagination", { pagination: { pageIndex: 1, pageSize: 50 } }],
      ["startTime", { startTime: "2025-02-02T00:00:00" }],
      ["columnFilters", { columnFilters: [{ id: LOG_FILTER_IDS.TEAM_ID, value: "team-2" }] }],
    ])("refetches when %s changes", async (_label, nextProps) => {
      const { rerender } = renderHook((props: HookOverrides) => useLogFilterLogic({ ...defaultProps, ...props }), {
        wrapper,
        initialProps: {},
      });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(1));
      rerender(nextProps);
      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalledTimes(2));
    });
  });

  describe("query enablement", () => {
    it("does not query when the request logs tab is inactive", async () => {
      renderFilterHook({ activeTab: "audit logs" });

      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(uiSpendLogsCall).not.toHaveBeenCalled();
    });

    it("does not query when credentials are missing", async () => {
      renderFilterHook({ accessToken: null });

      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(uiSpendLogsCall).not.toHaveBeenCalled();
    });
  });

  describe("filterByCurrentUser", () => {
    it("scopes to the current user when no explicit user filter is set", async () => {
      renderFilterHook({ filterByCurrentUser: true });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()?.params).toMatchObject({ user_id: "user-1" });
    });

    it("lets an explicit user filter win over the current-user scope", async () => {
      renderFilterHook({
        filterByCurrentUser: true,
        columnFilters: [{ id: LOG_FILTER_IDS.USER_ID, value: "someone-else" }],
      });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()?.params).toMatchObject({ user_id: "someone-else" });
    });
  });

  it("returns an empty payload and does not crash when the call fails", async () => {
    vi.mocked(uiSpendLogsCall).mockRejectedValue(new Error("boom"));
    const { result } = renderFilterHook();

    await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
    expect(result.current.filteredLogs.data).toEqual([]);
    expect(result.current.filteredLogs.total).toBe(0);
  });
});

describe("getFilterValue", () => {
  it("trims values and treats blank ones as absent", () => {
    const filters: ColumnFiltersState = [
      { id: "team_id", value: "  team-1  " },
      { id: "key_hash", value: "   " },
      { id: "status", value: 42 },
    ];

    expect(getFilterValue(filters, "team_id")).toBe("team-1");
    expect(getFilterValue(filters, "key_hash")).toBeUndefined();
    expect(getFilterValue(filters, "status")).toBeUndefined();
    expect(getFilterValue(filters, "missing")).toBeUndefined();
  });
});

describe("getLiveTailRefetchInterval", () => {
  it("polls every 15s when live tail is on and on the first page", () => {
    expect(getLiveTailRefetchInterval(true, 0)).toBe(LIVE_TAIL_INTERVAL_MS);
  });

  it("does not poll when live tail is off", () => {
    expect(getLiveTailRefetchInterval(false, 0)).toBe(false);
  });

  it("does not poll past the first page, even with live tail on", () => {
    expect(getLiveTailRefetchInterval(true, 1)).toBe(false);
  });
});
