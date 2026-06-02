import moment from "moment";
import { useCallback, useDeferredValue, useEffect, useMemo, useState } from "react";
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
import { useQuery } from "@tanstack/react-query";
import { Switch } from "antd";
import { internalUserRoles } from "../../utils/roles";
import DeletedKeysPage from "../DeletedKeysPage/DeletedKeysPage";
import DeletedTeamsPage from "../DeletedTeamsPage/DeletedTeamsPage";
import { KeyResponse } from "../key_team_helpers/key_list";
import FilterComponent from "../molecules/filter";
import { errorStatsCall, failureLogsAnalyticsPaginatedCall, keyInfoV1Call } from "../networking";
import KeyInfoView from "../templates/key_info_view";
import AuditLogs from "./audit_logs";
import { createColumns, LogEntry, type LogsSortField } from "./columns";
import { AGENT_CALL_TYPES, MCP_CALL_TYPES } from "./constants";
import { ErrorStatsTable } from "./ErrorStatsTable";
import { getLogFilterOptions } from "./filter_options";
import { useLogFilterLogic, defaultFilters, FILTER_KEYS, type LogFilterState } from "./log_filter_logic";
import { LogDetailsDrawer } from "./LogDetailsDrawer";
import { LogsTableToolbar } from "./LogsTableToolbar";
import { DataTable } from "./table";
import { AntDLoadingSpinner } from "../ui/AntDLoadingSpinner";
import ConcurrentRequestLogs from "./concurrent_request_logs";

interface SpendLogsTableProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
}

interface ErrorStatsResponse {
  time_bucket_size: string;
  data: Array<{
    time_bucket: string;
    extracted_error: string;
    count: number;
  }>;
}

interface FailureLogsAnalyticsResponse {
  data: LogEntry[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

const emptyFailureLogsAnalytics: FailureLogsAnalyticsResponse = {
  data: [],
  total: 0,
  page: 1,
  page_size: 50,
  total_pages: 0,
};

export default function SpendLogsTable({ accessToken, token, userRole, userID, premiumUser }: SpendLogsTableProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(50);

  const [startTime, setStartTime] = useState<string>(moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm"));
  const [endTime, setEndTime] = useState<string>(moment().format("YYYY-MM-DDTHH:mm"));

  const [isCustomDate, setIsCustomDate] = useState(false);
  const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
  const [selectedKeyInfo, setSelectedKeyInfo] = useState<KeyResponse | null>(null);
  const [selectedKeyIdInfoView, setSelectedKeyIdInfoView] = useState<string | null>(null);
  const [filterByCurrentUser, setFilterByCurrentUser] = useState(userRole && internalUserRoles.includes(userRole));
  const [activeTab, setActiveTab] = useState("request logs");
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [selectedErrorCategories, setSelectedErrorCategories] = useState<string[]>([]);
  const [failureLogsAnalyticsCurrentPage, setFailureLogsAnalyticsCurrentPage] = useState(1);
  const [failureLogsAnalyticsCurrentPageSize] = useState(50);

  const [selectedLog, setSelectedLog] = useState<LogEntry | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  const [sortBy, setSortBy] = useState<LogsSortField>("startTime");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  const [selectedTimeInterval, setSelectedTimeInterval] = useState<{ value: number; unit: string }>({
    value: 24,
    unit: "hours",
  });

  const [isLiveTail, setIsLiveTail] = useState<boolean>(() => {
    const storedValue = sessionStorage.getItem("isLiveTail");
    return storedValue !== null ? JSON.parse(storedValue) : true;
  });

  useEffect(() => {
    sessionStorage.setItem("isLiveTail", JSON.stringify(isLiveTail));
  }, [isLiveTail]);

  // Timestamp for forcing FilterComponent remount during live tail
  const [liveTailTimestamp, setLiveTailTimestamp] = useState<number>(Date.now());

  useEffect(() => {
    if (isLiveTail && !isCustomDate) {
      const interval = setInterval(() => {
        setLiveTailTimestamp(Date.now());
      }, 15000);
      return () => clearInterval(interval);
    }
  }, [isLiveTail, isCustomDate]);

  useEffect(() => {
    const fetchKeyInfo = async () => {
      if (selectedKeyIdInfoView && accessToken) {
        const keyData = await keyInfoV1Call(accessToken, selectedKeyIdInfoView);

        const keyResponse: KeyResponse = {
          ...keyData["info"],
          token: selectedKeyIdInfoView,
          api_key: selectedKeyIdInfoView,
        };
        setSelectedKeyInfo(keyResponse);
      }
    };
    fetchKeyInfo();
  }, [selectedKeyIdInfoView, accessToken]);

  useEffect(() => {
    if (userRole && internalUserRoles.includes(userRole)) {
      setFilterByCurrentUser(true);
    }
  }, [userRole]);

  const {
    logsQuery,
    filteredLogs,
    allTeams,
    handleFilterChange,
    handleFilterReset: handleFilterResetFromHook,
  } = useLogFilterLogic({
    accessToken,
    token,
    userRole,
    userID,
    filters,
    setFilters,
    filterByCurrentUser: !!filterByCurrentUser,
    activeTab,
    isLiveTail,
    startTime,
    endTime,
    pageSize,
    isCustomDate,
    setCurrentPage,
    sortBy,
    sortOrder,
    currentPage,
  });

  const errorStatsQuery = useQuery<ErrorStatsResponse>({
    queryKey: ["errorStats", startTime, endTime, isCustomDate, filters, filterByCurrentUser ? userID : null, activeTab],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        return { time_bucket_size: "", data: [] };
      }

      const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTime = isCustomDate
        ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
        : moment().utc().format("YYYY-MM-DD HH:mm:ss");

      return (
        (await errorStatsCall(
          accessToken,
          filters[FILTER_KEYS.KEY_HASH] || undefined,
          filters[FILTER_KEYS.TEAM_ID] || undefined,
          filters[FILTER_KEYS.REQUEST_ID] || undefined,
          formattedStartTime,
          formattedEndTime,
          filters[FILTER_KEYS.USER_ID] || (filterByCurrentUser ? userID ?? undefined : undefined),
          filters[FILTER_KEYS.END_USER] || undefined,
          filters[FILTER_KEYS.STATUS] || undefined,
          filters[FILTER_KEYS.PUBLIC_MODEL_OR_SEARCH_TOOL] || undefined,
          filters[FILTER_KEYS.MODEL] || undefined,
          filters[FILTER_KEYS.KEY_ALIAS] || undefined,
          filters[FILTER_KEYS.ERROR_CODE] || undefined,
          filters[FILTER_KEYS.ERROR_MESSAGE] || undefined,
        )) || { time_bucket_size: "", data: [] }
      );
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && activeTab === "request logs" && showAnalytics,
    refetchInterval: isLiveTail && !isCustomDate ? 15000 : false,
    refetchIntervalInBackground: false,
  });

  const failureLogsAnalytics = useQuery<FailureLogsAnalyticsResponse>({
    queryKey: [
      "failureLogsAnalytics",
      startTime,
      endTime,
      isCustomDate,
      filters,
      filterByCurrentUser ? userID : null,
      selectedErrorCategories,
      failureLogsAnalyticsCurrentPage,
      failureLogsAnalyticsCurrentPageSize,
      activeTab,
    ],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        return emptyFailureLogsAnalytics;
      }

      const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTime = isCustomDate
        ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
        : moment().utc().format("YYYY-MM-DD HH:mm:ss");

      return (
        (await failureLogsAnalyticsPaginatedCall(accessToken, {
          api_key: filters[FILTER_KEYS.KEY_HASH] || undefined,
          team_id: filters[FILTER_KEYS.TEAM_ID] || undefined,
          request_id: filters[FILTER_KEYS.REQUEST_ID] || undefined,
          start_date: formattedStartTime,
          end_date: formattedEndTime,
          user_id: filters[FILTER_KEYS.USER_ID] || (filterByCurrentUser ? userID ?? undefined : undefined),
          end_user: filters[FILTER_KEYS.END_USER] || undefined,
          model: filters[FILTER_KEYS.PUBLIC_MODEL_OR_SEARCH_TOOL] || undefined,
          model_id: filters[FILTER_KEYS.MODEL] || undefined,
          key_alias: filters[FILTER_KEYS.KEY_ALIAS] || undefined,
          error_code: filters[FILTER_KEYS.ERROR_CODE] || undefined,
          error_message: filters[FILTER_KEYS.ERROR_MESSAGE] || undefined,
          error_classes: selectedErrorCategories.length > 0 ? selectedErrorCategories.join(",") : undefined,
          page: failureLogsAnalyticsCurrentPage,
          page_size: failureLogsAnalyticsCurrentPageSize,
        })) || emptyFailureLogsAnalytics
      );
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && activeTab === "request logs" && showAnalytics,
    refetchInterval: isLiveTail && !isCustomDate ? 15000 : false,
    refetchIntervalInBackground: false,
  });

  useEffect(() => {
    setFailureLogsAnalyticsCurrentPage(1);
  }, [selectedErrorCategories, startTime, endTime, filters]);

  const handleFilterReset = useCallback(() => {
    handleFilterResetFromHook();
    setStartTime(moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm"));
    setEndTime(moment().format("YYYY-MM-DDTHH:mm"));
    setIsCustomDate(false);
    setSelectedTimeInterval({ value: 24, unit: "hours" });
    setSelectedErrorCategories([]);
    setFailureLogsAnalyticsCurrentPage(1);
    setCurrentPage(1);
  }, [handleFilterResetFromHook]);

  const handleSortChange = useCallback((newSortBy: LogsSortField, newSortOrder: "asc" | "desc") => {
    setSortBy(newSortBy);
    setSortOrder(newSortOrder);
    setCurrentPage(1);
  }, []);

  const columns = useMemo(
    () => createColumns({ sortBy, sortOrder, onSortChange: handleSortChange }),
    [sortBy, sortOrder, handleSortChange],
  );

  const filteredData = useMemo(() => {
    const searchedLogs = filteredLogs.data.filter((log) => {
      const matchesSearch =
        !searchTerm ||
        log.request_id.includes(searchTerm) ||
        log.model.includes(searchTerm) ||
        (log.user && log.user.includes(searchTerm));

      return matchesSearch;
    });

    const sessionCompositionById = searchedLogs.reduce<Record<string, { llm: number; agent: number; mcp: number }>>(
      (acc, log) => {
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
      },
      {},
    );

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
          request_duration_ms: log.request_duration_ms,
          session_llm_count: sessionComposition?.llm ?? undefined,
          session_mcp_count: sessionComposition?.mcp ?? undefined,
          session_agent_count: sessionComposition?.agent ?? undefined,
          onKeyHashClick: (keyHash: string) => setSelectedKeyIdInfoView(keyHash),
          onSessionClick: (sessionId: string) => {
            if (sessionId) {
              setSelectedSessionId(sessionId);
              setSelectedLog(log);
              setIsDrawerOpen(true);
            }
          },
        };
      })
      .filter((log) => {
        if (!log.session_id || (log.session_total_count || 1) <= 1) return true;
        return sessionRepresentativeMap.get(log.session_id)?.requestId === log.request_id;
      });
  }, [filteredLogs.data, searchTerm]);

  const deferredData = useDeferredValue(filteredData);
  const isStale = deferredData !== filteredData;
  const isButtonLoading = logsQuery.isFetching || isStale || (showAnalytics && failureLogsAnalytics.isFetching);
  const isRefiltering = logsQuery.isPlaceholderData;
  const isLogsLoading = logsQuery.isLoading || isRefiltering;
  const failureLogsData = failureLogsAnalytics.data ?? emptyFailureLogsAnalytics;
  const drawerLogs = showAnalytics ? failureLogsData.data : filteredData;

  if (!accessToken || !token || !userRole || !userID) {
    return (
      <div className="flex items-center justify-center h-64">
        <AntDLoadingSpinner size="large" />
      </div>
    );
  }

  const handleRowClick = (log: LogEntry) => {
    if (log.session_id && (log.session_total_count || 1) > 1) {
      setSelectedSessionId(log.session_id);
      setSelectedLog(log);
      setIsDrawerOpen(true);
      return;
    }
    setSelectedSessionId(null);
    setSelectedLog(log);
    setIsDrawerOpen(true);
  };

  return (
    <div className="w-full p-6 overflow-x-hidden box-border">
      <TabGroup
        defaultIndex={0}
        onIndexChange={(index) =>
          setActiveTab(
            ["request logs", "audit logs", "deleted keys", "deleted teams", "concurrent request logs"][index] ??
              "request logs",
          )
        }
      >
        <TabList>
          <Tab>Request Logs</Tab>
          <Tab>Audit Logs</Tab>
          <Tab>Deleted Keys</Tab>
          <Tab>Deleted Teams</Tab>
          <Tab>Concurrent Request Logs</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <div className="flex items-center justify-between mb-4">
              <h1 className="text-xl font-semibold">Request Logs</h1>
            </div>
            {selectedKeyInfo && selectedKeyIdInfoView && selectedKeyInfo.api_key === selectedKeyIdInfoView ? (
              <KeyInfoView
                keyId={selectedKeyIdInfoView}
                keyData={selectedKeyInfo}
                teams={allTeams ?? []}
                onClose={() => setSelectedKeyIdInfoView(null)}
                backButtonText="Back to Logs"
              />
            ) : (
              <>
                <FilterComponent
                  key={`${startTime}-${endTime}-${isLiveTail && !isCustomDate ? liveTailTimestamp : "static"}`}
                  options={getLogFilterOptions(accessToken)}
                  onApplyFilters={handleFilterChange}
                  onResetFilters={handleFilterReset}
                  initialValues={filters}
                  initialShowFilters={showFilters}
                  onShowFiltersChange={setShowFilters}
                />
                <div className="bg-white rounded-lg shadow-sm w-full max-w-full box-border">
                  <LogsTableToolbar
                    searchTerm={searchTerm}
                    onSearchChange={setSearchTerm}
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
                    currentPage={currentPage}
                    onCurrentPageChange={setCurrentPage}
                    pageSize={pageSize}
                    isLoading={isLogsLoading}
                    isButtonLoading={isButtonLoading}
                    onRefetch={() => {
                      logsQuery.refetch();
                      if (showAnalytics) {
                        errorStatsQuery.refetch();
                        failureLogsAnalytics.refetch();
                      }
                    }}
                    filteredLogs={filteredLogs}
                    showPagination={!showAnalytics}
                  />
                  <div className="border-b px-6 py-3 flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-900">Analytics</span>
                    <Switch checked={showAnalytics} onChange={setShowAnalytics} />
                  </div>
                  {showAnalytics ? (
                    <>
                      <ErrorStatsTable
                        data={errorStatsQuery.data?.data || []}
                        timeBucketSize={errorStatsQuery.data?.time_bucket_size}
                        onTimeRangeSelect={(selectedStartTime, selectedEndTime) => {
                          setStartTime(selectedStartTime);
                          setEndTime(selectedEndTime);
                          setFailureLogsAnalyticsCurrentPage(1);
                        }}
                        setCurrentPage={setCurrentPage}
                        setIsCustomDate={setIsCustomDate}
                        onSelectedCategoriesChange={setSelectedErrorCategories}
                      />
                      <div className="mt-6 px-4 pb-4">
                        {failureLogsData.total > 0 && (
                          <div className="mb-4 flex flex-wrap items-center justify-between gap-3 px-2">
                            <h2 className="text-lg font-semibold">Failure Logs</h2>
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-sm text-gray-700">
                                Page {failureLogsAnalytics.isLoading ? "..." : failureLogsAnalyticsCurrentPage} of{" "}
                                {failureLogsAnalytics.isLoading ? "..." : failureLogsData.total_pages}
                              </span>
                              <span className="text-sm text-gray-700">
                                Showing{" "}
                                {failureLogsAnalytics.isLoading
                                  ? "..."
                                  : (failureLogsAnalyticsCurrentPage - 1) * failureLogsAnalyticsCurrentPageSize +
                                    1}{" "}
                                -{" "}
                                {failureLogsAnalytics.isLoading
                                  ? "..."
                                  : Math.min(
                                      failureLogsAnalyticsCurrentPage * failureLogsAnalyticsCurrentPageSize,
                                      failureLogsData.total,
                                    )}{" "}
                                of {failureLogsAnalytics.isLoading ? "..." : failureLogsData.total} results
                              </span>
                              <button
                                onClick={() => setFailureLogsAnalyticsCurrentPage((page) => Math.max(1, page - 1))}
                                disabled={failureLogsAnalytics.isLoading || failureLogsAnalyticsCurrentPage === 1}
                                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                Previous
                              </button>
                              <button
                                onClick={() =>
                                  setFailureLogsAnalyticsCurrentPage((page) =>
                                    Math.min(failureLogsData.total_pages || 1, page + 1),
                                  )
                                }
                                disabled={
                                  failureLogsAnalytics.isLoading ||
                                  failureLogsAnalyticsCurrentPage === (failureLogsData.total_pages || 1)
                                }
                                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                Next
                              </button>
                            </div>
                          </div>
                        )}
                        <DataTable
                          columns={columns}
                          data={failureLogsData.data}
                          onRowClick={handleRowClick}
                          isLoading={failureLogsAnalytics.isLoading}
                        />
                      </div>
                    </>
                  ) : (
                    <DataTable
                      columns={columns}
                      data={deferredData}
                      onRowClick={handleRowClick}
                      isLoading={isLogsLoading}
                    />
                  )}
                </div>
              </>
            )}
          </TabPanel>
          <TabPanel>
            <AuditLogs
              userID={userID}
              userRole={userRole}
              token={token}
              accessToken={accessToken}
              isActive={activeTab === "audit logs"}
              premiumUser={premiumUser}
            />
          </TabPanel>
          <TabPanel>
            <DeletedKeysPage />
          </TabPanel>
          <TabPanel>
            <DeletedTeamsPage />
          </TabPanel>
          <TabPanel>
            <ConcurrentRequestLogs accessToken={accessToken} />
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <LogDetailsDrawer
        open={isDrawerOpen}
        onClose={() => {
          setIsDrawerOpen(false);
          setSelectedSessionId(null);
        }}
        logEntry={selectedLog}
        sessionId={selectedSessionId}
        accessToken={accessToken}
        allLogs={drawerLogs}
        onSelectLog={setSelectedLog}
        startTime={moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss")}
      />
    </div>
  );
}
