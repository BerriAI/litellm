import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ColumnFiltersState, PaginationState, SortingState } from "@tanstack/react-table";
import { renderHook, waitFor } from "@testing-library/react";
import moment from "moment";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_LOGS_SORTING,
  formatLogsWindow,
  getFilterValue,
  getLiveTailRefetchInterval,
  getLogsWindowEndBound,
  LIVE_TAIL_INTERVAL_MS,
  LOG_FILTER_IDS,
  LOGS_WINDOW_TICK_MS,
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
import { fetchAllTeams } from "@/components/key_team_helpers/filter_helpers";
import type { Team } from "../key_team_helpers/key_list";

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
  activeTab: "request logs",
  isLiveTail: false,
  excludeInternalHealthChecks: false,
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
      { id: LOG_FILTER_IDS.CACHE_STATUS, value: "hit", param: "cache_hit_filter" },
      { id: LOG_FILTER_IDS.CACHE_STATUS, value: "miss", param: "cache_hit_filter" },
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
      ["excludeInternalHealthChecks", { excludeInternalHealthChecks: true }],
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

  describe("hide health checks toggle", () => {
    it("passes exclude_internal_health_checks when the toggle is on", async () => {
      renderFilterHook({ excludeInternalHealthChecks: true });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()?.params).toMatchObject({ exclude_internal_health_checks: true });
    });

    it("passes exclude_internal_health_checks as false when the toggle is off", async () => {
      renderFilterHook();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()?.params).toMatchObject({ exclude_internal_health_checks: false });
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

  describe("user scope", () => {
    it("leaves an empty user filter for the backend to authorize", async () => {
      renderFilterHook();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()?.params?.user_id).toBeUndefined();
    });

    it("sends an explicit user filter for the backend to intersect with authorization", async () => {
      renderFilterHook({
        columnFilters: [{ id: LOG_FILTER_IDS.USER_ID, value: "someone-else" }],
      });

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCallParams()?.params).toMatchObject({ user_id: "someone-else" });
    });
  });

  describe("team filter list scope", () => {
    const callerTeams = [{ team_id: "team-a" }, { team_id: "team-b" }] as Team[];

    it("scopes /team/list to an internal user and still surfaces their teams", async () => {
      vi.mocked(fetchAllTeams).mockResolvedValue(callerTeams);

      const { result } = renderFilterHook({ userRole: "Internal User", userID: "member-7" });

      await waitFor(() => expect(fetchAllTeams).toHaveBeenCalled());
      expect(fetchAllTeams).toHaveBeenCalledWith("test-token", null, "member-7");
      await waitFor(() => expect(result.current.allTeams).toEqual(callerTeams));
    });

    it("scopes /team/list for an internal viewer", async () => {
      vi.mocked(fetchAllTeams).mockResolvedValue(callerTeams);

      renderFilterHook({ userRole: "Internal Viewer", userID: "member-7" });

      await waitFor(() => expect(fetchAllTeams).toHaveBeenCalledWith("test-token", null, "member-7"));
    });

    it.each(["Admin", "Admin Viewer", "Org Admin"])(
      "leaves /team/list unscoped for %s so the broad list survives",
      async (userRole) => {
        vi.mocked(fetchAllTeams).mockResolvedValue(callerTeams);

        renderFilterHook({ userRole, userID: "member-7" });

        await waitFor(() => expect(fetchAllTeams).toHaveBeenCalledWith("test-token", null, null));
      },
    );
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

describe("formatLogsWindow", () => {
  it("pins the end bound for a custom range", () => {
    const w = formatLogsWindow("2026-07-23T00:00", "2026-07-24T06:00", true);

    expect(w.start_date).toBe(moment("2026-07-23T00:00").utc().format("YYYY-MM-DD HH:mm:ss"));
    expect(w.end_date).toBe(moment("2026-07-24T06:00").utc().format("YYYY-MM-DD HH:mm:ss"));
  });

  it("ends a preset range at now, not at the stored end time", () => {
    const w = formatLogsWindow("2026-07-23T00:00", "1999-01-01T00:00", false);

    expect(w.end_date > "2020-01-01 00:00:00").toBe(true);
  });
});

describe("getLogsWindowEndBound", () => {
  const BUCKET_START = 16666 * LOGS_WINDOW_TICK_MS;

  it("holds steady inside a bucket so a memoized window does not refetch per render", () => {
    expect(getLogsWindowEndBound(BUCKET_START)).toBe(getLogsWindowEndBound(BUCKET_START + LOGS_WINDOW_TICK_MS - 1));
  });

  it("advances once the bucket rolls over so a preset window follows the table", () => {
    expect(getLogsWindowEndBound(BUCKET_START + LOGS_WINDOW_TICK_MS)).toBe(
      getLogsWindowEndBound(BUCKET_START) + LOGS_WINDOW_TICK_MS,
    );
  });

  it("never trails the fetch it was derived from", () => {
    // Trailing is the live-tail bug: the table shows rows the filter window excludes.
    for (const offset of [0, 1, LOGS_WINDOW_TICK_MS - 1]) {
      expect(getLogsWindowEndBound(BUCKET_START + offset)).toBeGreaterThan(BUCKET_START + offset);
    }
  });

  it("advances at least once per live-tail refetch interval", () => {
    expect(LOGS_WINDOW_TICK_MS).toBeLessThanOrEqual(4 * LIVE_TAIL_INTERVAL_MS);
  });
});

describe("formatLogsWindow preset end bound", () => {
  it("uses the supplied bound for a preset range so callers can memoize it", () => {
    const bound = Date.UTC(2026, 6, 24, 12, 0, 0);

    expect(formatLogsWindow("2026-07-23T00:00", "1999-01-01T00:00", false, bound).end_date).toBe(
      moment(bound).utc().format("YYYY-MM-DD HH:mm:ss"),
    );
  });

  it("ignores the supplied bound for a custom range", () => {
    const bound = Date.UTC(2026, 6, 24, 12, 0, 0);

    expect(formatLogsWindow("2026-07-23T00:00", "2026-07-24T06:00", true, bound).end_date).toBe(
      moment("2026-07-24T06:00").utc().format("YYYY-MM-DD HH:mm:ss"),
    );
  });
});
