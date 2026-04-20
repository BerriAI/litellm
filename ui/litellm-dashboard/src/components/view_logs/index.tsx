import moment from "moment";
import { useCallback, useEffect, useMemo, useState } from "react";
import { SettingOutlined } from "@ant-design/icons";
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
import { Button } from "antd";
import { internalUserRoles } from "../../utils/roles";
import DeletedKeysPage from "../DeletedKeysPage/DeletedKeysPage";
import DeletedTeamsPage from "../DeletedTeamsPage/DeletedTeamsPage";
import { KeyResponse } from "../key_team_helpers/key_list";
import FilterComponent from "../molecules/filter";
import { keyInfoV1Call } from "../networking";
import KeyInfoView from "../templates/key_info_view";
import AuditLogs from "./audit_logs";
import { createColumns, LogEntry, type LogsSortField } from "./columns";
import { AGENT_CALL_TYPES, MCP_CALL_TYPES } from "./constants";
import { getLogFilterOptions } from "./filter_options";
import { useLogFilterLogic, defaultFilters, type LogFilterState } from "./log_filter_logic";
import { LogDetailsDrawer } from "./LogDetailsDrawer";
import { LogsTableToolbar } from "./LogsTableToolbar";
import SpendLogsSettingsModal from "./SpendLogsSettingsModal/SpendLogsSettingsModal";
import { DataTable } from "./table";

interface SpendLogsTableProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
}

export default function SpendLogsTable({
  accessToken,
  token,
  userRole,
  userID,
  premiumUser,
}: SpendLogsTableProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(50);

  // New state variables for Start and End Time
  const [startTime, setStartTime] = useState<string>(moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm"));
  const [endTime, setEndTime] = useState<string>(moment().format("YYYY-MM-DDTHH:mm"));

  const [isCustomDate, setIsCustomDate] = useState(false);
  const [filters, setFilters] = useState<LogFilterState>(defaultFilters);
  const [selectedKeyInfo, setSelectedKeyInfo] = useState<KeyResponse | null>(null);
  const [selectedKeyIdInfoView, setSelectedKeyIdInfoView] = useState<string | null>(null);
  const [filterByCurrentUser, setFilterByCurrentUser] = useState(userRole && internalUserRoles.includes(userRole));
  const [activeTab, setActiveTab] = useState("request logs");

  const [selectedLog, setSelectedLog] = useState<LogEntry | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [isSpendLogsSettingsModalVisible, setIsSpendLogsSettingsModalVisible] = useState(false);

  const [sortBy, setSortBy] = useState<LogsSortField>("startTime");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");


  const [selectedTimeInterval, setSelectedTimeInterval] = useState<{ value: number; unit: string }>({
    value: 24,
    unit: "hours",
  });

  const [isLiveTail, setIsLiveTail] = useState<boolean>(() => {
    const storedValue = sessionStorage.getItem("isLiveTail");
    // default to true if nothing is stored
    return storedValue !== null ? JSON.parse(storedValue) : true;
  });

  useEffect(() => {
    sessionStorage.setItem("isLiveTail", JSON.stringify(isLiveTail));
  }, [isLiveTail]);


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

  const handleFilterReset = useCallback(() => {
    handleFilterResetFromHook();
    setStartTime(moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm"));
    setEndTime(moment().format("YYYY-MM-DDTHH:mm"));
    setIsCustomDate(false);
    setSelectedTimeInterval({ value: 24, unit: "hours" });
    setCurrentPage(1);
  }, [handleFilterResetFromHook]);

  const handleSortChange = useCallback(
    (newSortBy: LogsSortField, newSortOrder: "asc" | "desc") => {
      setSortBy(newSortBy);
      setSortOrder(newSortOrder);
      setCurrentPage(1);
    },
    [],
  );

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

      // No need for additional filtering since we're now handling this in the API call
      return matchesSearch;
    });

    const sessionCompositionById = searchedLogs.reduce<Record<string, { llm: number; agent: number; mcp: number }>>((acc, log) => {
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

    // Build a single-pass map of session_id → representative request_id.
    // Prefers an LLM row over an MCP row as the representative.
    const sessionRepresentativeMap = new Map<string, { requestId: string; isMcp: boolean }>();
    for (const log of searchedLogs) {
      if (!log.session_id || (log.session_total_count || 1) <= 1) continue;
      const isMcp = MCP_CALL_TYPES.includes(log.call_type);
      const existing = sessionRepresentativeMap.get(log.session_id);
      if (!existing || (existing.isMcp && !isMcp)) {
        sessionRepresentativeMap.set(log.session_id, { requestId: log.request_id, isMcp });
      }
    }

    return (
      searchedLogs
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
        // Deduplicate multi-call sessions using the pre-built map (O(1) per row).
        .filter((log) => {
          if (!log.session_id || (log.session_total_count || 1) <= 1) return true;
          return sessionRepresentativeMap.get(log.session_id)?.requestId === log.request_id;
        })
    );
  }, [filteredLogs.data, searchTerm]);

  if (!accessToken || !token || !userRole || !userID) {
    return null;
  }

  const handleRowClick = (log: LogEntry) => {
    // Multi-call session row: open in the same right-side drawer (session mode)
    if (log.session_id && (log.session_total_count || 1) > 1) {
      setSelectedSessionId(log.session_id);
      setSelectedLog(log);
      setIsDrawerOpen(true);
      return;
    }
    // Single-call row: open the detail drawer
    setSelectedSessionId(null);
    setSelectedLog(log);
    setIsDrawerOpen(true);
  };

  return (
    <div className="w-full max-w-screen p-6 overflow-x-hidden box-border">
      <TabGroup defaultIndex={0} onIndexChange={(index) => setActiveTab(index === 0 ? "request logs" : "audit logs")}>
        <TabList>
          <Tab>Request Logs</Tab>
          <Tab>Audit Logs</Tab>
          <Tab>Deleted Keys</Tab>
          <Tab>Deleted Teams</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <div className="flex items-center justify-between mb-4">
              <h1 className="text-xl font-semibold">Request Logs</h1>
              <Button
                icon={<SettingOutlined />}
                onClick={() => setIsSpendLogsSettingsModalVisible(true)}
                title="Spend Logs Settings"
              />
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
                  options={getLogFilterOptions(accessToken)}
                  onApplyFilters={handleFilterChange}
                  onResetFilters={handleFilterReset}
                />
                <SpendLogsSettingsModal
                  isVisible={isSpendLogsSettingsModalVisible}
                  onCancel={() => setIsSpendLogsSettingsModalVisible(false)}
                  onSuccess={() => setIsSpendLogsSettingsModalVisible(false)}
                />
                <div className="bg-white rounded-lg shadow w-full max-w-full box-border">
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
                    isLoading={logsQuery.isLoading}
                    isButtonLoading={logsQuery.isFetching}
                    onRefetch={() => logsQuery.refetch()}
                    filteredLogs={filteredLogs}
                  />
                  <DataTable
                    columns={columns}
                    data={filteredData}
                    onRowClick={handleRowClick}
                    isLoading={logsQuery.isLoading}
                  />
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
          <TabPanel><DeletedKeysPage /></TabPanel>
          <TabPanel><DeletedTeamsPage /></TabPanel>
        </TabPanels>
      </TabGroup>

      {/* Log Details Drawer */}
      <LogDetailsDrawer
        open={isDrawerOpen}
        onClose={() => { setIsDrawerOpen(false); setSelectedSessionId(null); }}
        logEntry={selectedLog}
        sessionId={selectedSessionId}
        accessToken={accessToken}
        onOpenSettings={() => setIsSpendLogsSettingsModalVisible(true)}
        allLogs={filteredData}
        onSelectLog={setSelectedLog}
        startTime={moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss")}
      />
    </div>
  );
}

