"use client";

import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import {
  functionalUpdate,
  type ColumnFiltersState,
  type OnChangeFn,
  type PaginationState,
  type SortingState,
} from "@tanstack/react-table";
import moment from "moment";
import { parseAsBoolean, parseAsString, useQueryState } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";

import { DEFAULT_PAGE_SIZE_OPTIONS, useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { AutoRouterModelGroupsProvider } from "@/components/shared/table_cells";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import type { KeyResponse } from "../key_team_helpers/key_list";
import { keyInfoV1Call, uiSpendLogsCall } from "../networking";
import KeyInfoView from "../templates/key_info_view";
import { LOGS_SORT_FIELD_MAP, type LogEntry } from "./columns";
import {
  DEFAULT_LOGS_SORTING,
  formatLogsWindow,
  getLogsWindowEndBound,
  LOG_FILTER_IDS,
  type PaginatedResponse,
  useLogFilterLogic,
} from "./log_filter_logic";
import { useLogDetailRouting } from "./logDetailRouting";
import { LogDetailsDrawer } from "./LogDetailsDrawer";
import { LiveTailBanner, LogsTableToolbar } from "./LogsTableToolbar";
import { RequestLogsTable } from "./RequestLogsTable";
import { CUSTOM_RANGE, useLogsTimeRange } from "./useLogsTimeRange";

const PAGE_SIZE = DEFAULT_PAGE_SIZE_OPTIONS[0];
const matchesLogId = (log: LogEntry, logId: string) => log.request_id === logId || log.litellm_call_id === logId;
const findLogById = (logs: readonly LogEntry[], logId: string): LogEntry | null =>
  logs.find((log) => log.request_id === logId) ?? logs.find((log) => log.litellm_call_id === logId) ?? null;

type LogFilterId = (typeof LOG_FILTER_IDS)[keyof typeof LOG_FILTER_IDS];
type FilterColumn = Exclude<LogFilterId, typeof LOG_FILTER_IDS.SEARCH>;

const isFilterColumn = (id: LogFilterId): id is FilterColumn => id !== LOG_FILTER_IDS.SEARCH;
const FILTER_COLUMNS: readonly FilterColumn[] = Object.values(LOG_FILTER_IDS).filter(isFilterColumn);

const TABLE_STATE_OPTIONS: UrlTableStateOptions<FilterColumn> = {
  sortFields: Object.keys(LOGS_SORT_FIELD_MAP),
  defaultSort: DEFAULT_LOGS_SORTING[0],
  defaultPageSize: PAGE_SIZE,
  filterColumns: FILTER_COLUMNS,
  urlKeys: {
    search: "log_search",
    filter_team_id: "filter_team",
    filter_cache_hit: "filter_cache",
    filter_user_id: "filter_user",
    filter_session_id: "filter_session",
  },
};

const withSearchFilter = (filters: ColumnFiltersState, search: string): ColumnFiltersState =>
  search === "" ? filters : [...filters, { id: LOG_FILTER_IDS.SEARCH, value: search }];

const searchFilterValue = (filters: ColumnFiltersState): string => {
  const value = filters.find((filter) => filter.id === LOG_FILTER_IDS.SEARCH)?.value;
  return typeof value === "string" ? value : "";
};

const tableRowCount = (response: PaginatedResponse, { pageIndex, pageSize }: PaginationState): number => {
  const pageRows = response.data.length;
  if (pageRows === 0 && pageIndex > 0) return response.total;
  const rowsThroughThisPage = pageIndex * pageSize + pageRows;
  const isLastPage = response.has_more === false || (response.has_more === undefined && pageRows < pageSize);
  return isLastPage ? rowsThroughThisPage : Math.max(response.total, rowsThroughThisPage);
};

interface RequestLogsPanelProps {
  accessToken: string;
  token: string;
  userRole: string;
  userID: string;
  isActive: boolean;
}

export default function RequestLogsPanel({ accessToken, token, userRole, userID, isActive }: RequestLogsPanelProps) {
  const {
    search,
    setSearch,
    sorting,
    onSortingChange,
    pagination,
    onPaginationChange,
    columnFilters: urlColumnFilters,
    onColumnFiltersChange,
  } = useUrlTableState(TABLE_STATE_OPTIONS);
  const [sessionCursors, setSessionCursors] = useState<Record<number, string>>({});
  const [excludeInternalHealthChecks, setExcludeInternalHealthChecks] = useQueryState(
    "hide_health_checks",
    parseAsBoolean.withDefault(false),
  );
  const [selectedKeyId, setSelectedKeyId] = useQueryState("key", parseAsString.withOptions({ history: "push" }));
  const [selectedLog, setSelectedLog] = useState<LogEntry | null>(null);

  const resetToFirstPage = useCallback(() => {
    setSessionCursors({});
    onPaginationChange((previous) => ({ ...previous, pageIndex: 0 }));
  }, [onPaginationChange]);

  const {
    timeRange,
    selectPreset,
    toggleCustomRange,
    setStartTime,
    setEndTime,
    reset: resetTimeRange,
  } = useLogsTimeRange(resetToFirstPage);
  const { startTime, endTime } = timeRange;
  const isCustomDate = timeRange.range === CUSTOM_RANGE;

  const {
    logId: urlLogId,
    sessionId: urlSessionId,
    openLog,
    openSession,
    selectLog,
    close: closeUrlLog,
  } = useLogDetailRouting();

  const [isLiveTail, setIsLiveTail] = useState<boolean>(() => {
    const storedValue = sessionStorage.getItem("isLiveTail");
    return storedValue !== null ? JSON.parse(storedValue) : true;
  });

  useEffect(() => {
    sessionStorage.setItem("isLiveTail", JSON.stringify(isLiveTail));
  }, [isLiveTail]);

  const columnFilters = useMemo(() => withSearchFilter(urlColumnFilters, search), [urlColumnFilters, search]);
  const [debouncedSearch] = useDebouncedValue(search, { wait: DEBOUNCE_WAIT_MS });
  const queryColumnFilters = useMemo(
    () => withSearchFilter(urlColumnFilters, debouncedSearch),
    [urlColumnFilters, debouncedSearch],
  );

  const { logsQuery, filteredLogs, allTeams, usesSessionCursor } = useLogFilterLogic({
    accessToken,
    token,
    userRole,
    userID,
    columnFilters: queryColumnFilters,
    activeTab: isActive ? "request logs" : "inactive",
    isLiveTail,
    excludeInternalHealthChecks,
    startTime,
    endTime,
    pagination,
    isCustomDate,
    sorting,
    sessionCursors,
  });

  // Follow the table's own last fetch so a live-tail refresh carries the filter
  // window with it; before the first fetch, fall back to the stored end time.
  const windowEndBound = getLogsWindowEndBound(logsQuery.dataUpdatedAt || Date.parse(endTime));
  const logsWindow = useMemo(
    () => formatLogsWindow(startTime, endTime, isCustomDate, windowEndBound),
    [startTime, endTime, isCustomDate, windowEndBound],
  );

  const keyInfoQueryOptions: UseQueryOptions<KeyResponse | null> = {
    queryKey: ["requestLogsKeyInfo", selectedKeyId, accessToken],
    queryFn: async () => {
      if (selectedKeyId === null) return null;
      const keyData = await keyInfoV1Call(accessToken, selectedKeyId);
      return {
        ...keyData["info"],
        token: selectedKeyId,
        api_key: selectedKeyId,
      };
    },
    enabled: selectedKeyId !== null,
  };

  const { data: selectedKeyInfo } = useQuery(keyInfoQueryOptions);

  const urlLogQueryOptions: UseQueryOptions<LogEntry | null> = {
    queryKey: ["logs", "byId", urlLogId, accessToken],
    queryFn: async () => {
      if (urlLogId === null) return null;
      const window = formatLogsWindow(startTime, endTime, isCustomDate);
      const response: PaginatedResponse = await uiSpendLogsCall({
        accessToken,
        start_date: window.start_date,
        end_date: window.end_date,
        page: 1,
        page_size: 1,
        params: { request_id: urlLogId },
      });
      return findLogById(response.data, urlLogId);
    },
    enabled: urlLogId !== null && !(selectedLog !== null && matchesLogId(selectedLog, urlLogId)),
    staleTime: Infinity,
  };

  const { data: urlLog } = useQuery(urlLogQueryOptions);

  const displayLog = useMemo<LogEntry | null>(() => {
    if (urlLogId === null) return null;
    if (selectedLog !== null && matchesLogId(selectedLog, urlLogId)) return selectedLog;
    return findLogById(filteredLogs.data, urlLogId) ?? urlLog ?? null;
  }, [urlLogId, selectedLog, filteredLogs.data, urlLog]);

  const displaySessionId = useMemo<string | null>(() => {
    if (urlSessionId !== null) return urlSessionId;
    if (displayLog?.session_id !== undefined && (displayLog.session_total_count || 1) > 1) {
      return displayLog.session_id;
    }
    return null;
  }, [urlSessionId, displayLog]);

  const isDrawerOpen = displayLog !== null || displaySessionId !== null;

  const rows: LogEntry[] = filteredLogs.data;
  const rowCount = tableRowCount(filteredLogs, pagination);

  const handleSearchChange = useCallback(
    (value: string) => {
      setSearch(value);
      setSessionCursors({});
    },
    [setSearch],
  );

  const handleSortingChange = useCallback<OnChangeFn<SortingState>>(
    (updaterOrValue) => {
      onSortingChange(updaterOrValue);
      setSessionCursors({});
    },
    [onSortingChange],
  );

  const handleColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>(
    (updaterOrValue) => {
      const next = functionalUpdate(updaterOrValue, columnFilters);
      onColumnFiltersChange(next.filter((filter) => filter.id !== LOG_FILTER_IDS.SEARCH));
      const nextSearch = searchFilterValue(next);
      if (nextSearch !== search) setSearch(nextSearch);
      setSessionCursors({});
    },
    [columnFilters, onColumnFiltersChange, search, setSearch],
  );

  const handlePaginationChange = useCallback<OnChangeFn<PaginationState>>(
    (updaterOrValue) => {
      const requested = functionalUpdate(updaterOrValue, pagination);
      if (!usesSessionCursor) {
        onPaginationChange(requested);
        return;
      }
      if (requested.pageSize !== pagination.pageSize) {
        setSessionCursors({});
        onPaginationChange({ ...requested, pageIndex: 0 });
        return;
      }
      if (requested.pageIndex !== pagination.pageIndex + 1) {
        onPaginationChange(requested);
        return;
      }
      const nextCursor = filteredLogs.next_session_cursor;
      if (!nextCursor || logsQuery.isPlaceholderData) return;
      setSessionCursors((previous) => ({ ...previous, [requested.pageIndex]: nextCursor }));
      onPaginationChange(requested);
    },
    [usesSessionCursor, pagination, onPaginationChange, filteredLogs.next_session_cursor, logsQuery.isPlaceholderData],
  );

  const handleExcludeInternalHealthChecksChange = useCallback(
    (value: boolean) => {
      void setExcludeInternalHealthChecks(value);
      resetToFirstPage();
    },
    [setExcludeInternalHealthChecks, resetToFirstPage],
  );

  const handleResetFilters = useCallback(() => {
    onColumnFiltersChange([]);
    setSearch("");
    resetTimeRange();
  }, [onColumnFiltersChange, setSearch, resetTimeRange]);

  const handleRowClick = useCallback(
    (log: LogEntry) => {
      setSelectedLog(log);
      if (log.session_id && (log.session_total_count || 1) > 1) {
        openSession(log.session_id, log.request_id);
      } else {
        openLog(log.request_id);
      }
    },
    [openLog, openSession],
  );

  const handleSessionClick = useCallback(
    (log: LogEntry) => {
      if (!log.session_id) return;
      setSelectedLog(log);
      openSession(log.session_id, log.request_id);
    },
    [openSession],
  );

  const handleSelectLog = useCallback(
    (log: LogEntry) => {
      setSelectedLog(log);
      selectLog(log.request_id, displaySessionId);
    },
    [selectLog, displaySessionId],
  );

  const handleKeyHashClick = useCallback(
    (keyHash: string) => {
      void setSelectedKeyId(keyHash);
    },
    [setSelectedKeyId],
  );

  if (selectedKeyInfo && selectedKeyId && selectedKeyInfo.api_key === selectedKeyId) {
    return (
      <KeyInfoView
        keyId={selectedKeyId}
        keyData={selectedKeyInfo}
        teams={allTeams ?? []}
        onClose={() => void setSelectedKeyId(null)}
        backButtonText="Back to Logs"
      />
    );
  }

  return (
    <AutoRouterModelGroupsProvider>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Request Logs</h1>
      </div>

      {isLiveTail && pagination.pageIndex === 0 && <LiveTailBanner onStop={() => setIsLiveTail(false)} />}

      <RequestLogsTable
        data={rows}
        rowCount={rowCount}
        isLoading={logsQuery.isPending}
        isError={logsQuery.isError}
        isRefreshing={logsQuery.isFetching}
        pagination={pagination}
        onPaginationChange={handlePaginationChange}
        sorting={sorting}
        onSortingChange={handleSortingChange}
        columnFilters={columnFilters}
        onColumnFiltersChange={handleColumnFiltersChange}
        searchValue={search}
        onSearchChange={handleSearchChange}
        onRefresh={() => void logsQuery.refetch()}
        onRowClick={handleRowClick}
        onKeyHashClick={handleKeyHashClick}
        onSessionClick={handleSessionClick}
        teams={allTeams ?? []}
        logsWindow={logsWindow}
        toolbarChildren={
          <LogsTableToolbar
            timeRange={timeRange}
            onPresetSelect={selectPreset}
            onCustomRangeToggle={toggleCustomRange}
            onStartTimeChange={setStartTime}
            onEndTimeChange={setEndTime}
            isLiveTail={isLiveTail}
            onIsLiveTailChange={setIsLiveTail}
            excludeInternalHealthChecks={excludeInternalHealthChecks}
            onExcludeInternalHealthChecksChange={handleExcludeInternalHealthChecksChange}
            onResetFilters={handleResetFilters}
          />
        }
      />

      <LogDetailsDrawer
        open={isDrawerOpen}
        onClose={closeUrlLog}
        logEntry={displayLog}
        sessionId={displaySessionId}
        accessToken={accessToken}
        allLogs={rows}
        onSelectLog={handleSelectLog}
        startTime={moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss")}
      />
    </AutoRouterModelGroupsProvider>
  );
}
