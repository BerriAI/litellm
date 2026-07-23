"use client";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import type { ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import moment from "moment";
import { useCallback, useEffect, useMemo, useState } from "react";

import { internalUserRoles } from "../../utils/roles";
import type { KeyResponse } from "../key_team_helpers/key_list";
import { keyInfoV1Call } from "../networking";
import KeyInfoView from "../templates/key_info_view";
import type { LogEntry } from "./columns";
import { AGENT_CALL_TYPES, MCP_CALL_TYPES } from "./constants";
import { DEFAULT_LOGS_SORTING, useLogFilterLogic } from "./log_filter_logic";
import { LogDetailsDrawer } from "./LogDetailsDrawer";
import { LiveTailBanner, LogsTableToolbar } from "./LogsTableToolbar";
import { RequestLogsTable } from "./RequestLogsTable";

const PAGE_SIZE = 50;
const DEFAULT_INTERVAL = { value: 24, unit: "hours" };

interface RequestLogsPanelProps {
  accessToken: string;
  token: string;
  userRole: string;
  userID: string;
  isActive: boolean;
}

interface SessionComposition {
  llm: number;
  agent: number;
  mcp: number;
}

export default function RequestLogsPanel({ accessToken, token, userRole, userID, isActive }: RequestLogsPanelProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: PAGE_SIZE });
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_LOGS_SORTING);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);

  const [startTime, setStartTime] = useState<string>(moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm"));
  const [endTime, setEndTime] = useState<string>(moment().format("YYYY-MM-DDTHH:mm"));
  const [isCustomDate, setIsCustomDate] = useState(false);
  const [selectedTimeInterval, setSelectedTimeInterval] = useState<{ value: number; unit: string }>(DEFAULT_INTERVAL);

  const [selectedKeyIdInfoView, setSelectedKeyIdInfoView] = useState<string | null>(null);
  const [selectedLog, setSelectedLog] = useState<LogEntry | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  const [isLiveTail, setIsLiveTail] = useState<boolean>(() => {
    const storedValue = sessionStorage.getItem("isLiveTail");
    return storedValue !== null ? JSON.parse(storedValue) : true;
  });

  useEffect(() => {
    sessionStorage.setItem("isLiveTail", JSON.stringify(isLiveTail));
  }, [isLiveTail]);

  const filterByCurrentUser = internalUserRoles.includes(userRole);

  const { logsQuery, filteredLogs, allTeams } = useLogFilterLogic({
    accessToken,
    token,
    userRole,
    userID,
    columnFilters,
    filterByCurrentUser,
    activeTab: isActive ? "request logs" : "inactive",
    isLiveTail,
    startTime,
    endTime,
    pagination,
    isCustomDate,
    sorting,
  });

  const keyInfoQueryOptions: UseQueryOptions<KeyResponse | null> = {
    queryKey: ["requestLogsKeyInfo", selectedKeyIdInfoView, accessToken],
    queryFn: async () => {
      if (selectedKeyIdInfoView === null) return null;
      const keyData = await keyInfoV1Call(accessToken, selectedKeyIdInfoView);
      return {
        ...keyData["info"],
        token: selectedKeyIdInfoView,
        api_key: selectedKeyIdInfoView,
      };
    },
    enabled: selectedKeyIdInfoView !== null,
  };

  const { data: selectedKeyInfo } = useQuery(keyInfoQueryOptions);

  const rows = useMemo<LogEntry[]>(() => {
    const searchedLogs = filteredLogs.data.filter((log) => {
      if (!searchTerm) return true;
      return (
        log.request_id.includes(searchTerm) ||
        log.model.includes(searchTerm) ||
        (log.user !== undefined && log.user.includes(searchTerm))
      );
    });

    const sessionCompositionById = searchedLogs.reduce<Record<string, SessionComposition>>((acc, log) => {
      if (!log.session_id) return acc;
      if (!acc[log.session_id]) {
        acc[log.session_id] = { llm: 0, agent: 0, mcp: 0 };
      }
      if (MCP_CALL_TYPES.includes(log.call_type)) {
        acc[log.session_id].mcp += 1;
      } else if (AGENT_CALL_TYPES.includes(log.call_type)) {
        acc[log.session_id].agent += 1;
      } else {
        acc[log.session_id].llm += 1;
      }
      return acc;
    }, {});

    const sessionRepresentativeMap = new Map<string, { requestId: string; isMcp: boolean }>();
    for (const log of searchedLogs) {
      if (!log.session_id || (log.session_total_count || 1) <= 1) continue;
      const isMcp = MCP_CALL_TYPES.includes(log.call_type);
      const existing = sessionRepresentativeMap.get(log.session_id);
      if (!existing || (existing.isMcp && !isMcp)) {
        sessionRepresentativeMap.set(log.session_id, { requestId: log.request_id, isMcp });
      }
    }

    return searchedLogs
      .map((log) => {
        const sessionComposition = log.session_id ? sessionCompositionById[log.session_id] : undefined;
        return {
          ...log,
          session_llm_count: sessionComposition?.llm ?? undefined,
          session_mcp_count: sessionComposition?.mcp ?? undefined,
          session_agent_count: sessionComposition?.agent ?? undefined,
        };
      })
      .filter((log) => {
        if (!log.session_id || (log.session_total_count || 1) <= 1) return true;
        return sessionRepresentativeMap.get(log.session_id)?.requestId === log.request_id;
      });
  }, [filteredLogs.data, searchTerm]);

  const handleSortingChange = useCallback<OnChangeFn<SortingState>>((updaterOrValue) => {
    setSorting(updaterOrValue);
    setPagination((previous) => ({ ...previous, pageIndex: 0 }));
  }, []);

  const handleColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>((updaterOrValue) => {
    setColumnFilters(updaterOrValue);
    setPagination((previous) => ({ ...previous, pageIndex: 0 }));
  }, []);

  const resetToFirstPage = useCallback(() => {
    setPagination((previous) => ({ ...previous, pageIndex: 0 }));
  }, []);

  const handleResetFilters = useCallback(() => {
    setColumnFilters([]);
    setSearchTerm("");
    setStartTime(moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm"));
    setEndTime(moment().format("YYYY-MM-DDTHH:mm"));
    setIsCustomDate(false);
    setSelectedTimeInterval(DEFAULT_INTERVAL);
    resetToFirstPage();
  }, [resetToFirstPage]);

  const handleRowClick = useCallback((log: LogEntry) => {
    const isMultiCallSession = log.session_id !== undefined && (log.session_total_count || 1) > 1;
    setSelectedSessionId(isMultiCallSession ? log.session_id ?? null : null);
    setSelectedLog(log);
    setIsDrawerOpen(true);
  }, []);

  const handleSessionClick = useCallback(
    (sessionId: string) => {
      if (!sessionId) return;
      const log = rows.find((candidate) => candidate.session_id === sessionId) ?? null;
      setSelectedSessionId(sessionId);
      setSelectedLog(log);
      setIsDrawerOpen(true);
    },
    [rows],
  );

  const handleKeyHashClick = useCallback((keyHash: string) => {
    setSelectedKeyIdInfoView(keyHash);
  }, []);

  if (selectedKeyInfo && selectedKeyIdInfoView && selectedKeyInfo.api_key === selectedKeyIdInfoView) {
    return (
      <KeyInfoView
        keyId={selectedKeyIdInfoView}
        keyData={selectedKeyInfo}
        teams={allTeams ?? []}
        onClose={() => setSelectedKeyIdInfoView(null)}
        backButtonText="Back to Logs"
      />
    );
  }

  return (
    <>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Request Logs</h1>
      </div>

      {isLiveTail && pagination.pageIndex === 0 && <LiveTailBanner onStop={() => setIsLiveTail(false)} />}

      <RequestLogsTable
        data={rows}
        rowCount={filteredLogs.total}
        isLoading={logsQuery.isLoading}
        isRefreshing={logsQuery.isFetching}
        pagination={pagination}
        onPaginationChange={setPagination}
        sorting={sorting}
        onSortingChange={handleSortingChange}
        columnFilters={columnFilters}
        onColumnFiltersChange={handleColumnFiltersChange}
        searchValue={searchTerm}
        onSearchChange={setSearchTerm}
        onRefresh={() => void logsQuery.refetch()}
        onRowClick={handleRowClick}
        onKeyHashClick={handleKeyHashClick}
        onSessionClick={handleSessionClick}
        teams={allTeams ?? []}
        accessToken={accessToken}
        toolbarChildren={
          <LogsTableToolbar
            startTime={startTime}
            onStartTimeChange={setStartTime}
            endTime={endTime}
            onEndTimeChange={setEndTime}
            isCustomDate={isCustomDate}
            onIsCustomDateChange={setIsCustomDate}
            selectedTimeInterval={selectedTimeInterval}
            onSelectedTimeIntervalChange={setSelectedTimeInterval}
            isLiveTail={isLiveTail}
            onIsLiveTailChange={setIsLiveTail}
            onResetToFirstPage={resetToFirstPage}
            onResetFilters={handleResetFilters}
          />
        }
      />

      <LogDetailsDrawer
        open={isDrawerOpen}
        onClose={() => {
          setIsDrawerOpen(false);
          setSelectedSessionId(null);
        }}
        logEntry={selectedLog}
        sessionId={selectedSessionId}
        accessToken={accessToken}
        allLogs={rows}
        onSelectLog={setSelectedLog}
        startTime={moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss")}
      />
    </>
  );
}
