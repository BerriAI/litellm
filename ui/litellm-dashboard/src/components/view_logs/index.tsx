import moment from "moment";
import { useQuery } from "@tanstack/react-query";
import { useState, useRef, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { uiSpendLogsCall, keyInfoV1Call, sessionSpendLogsCall } from "../networking";
import { DataTable } from "./table";
import { columns, LogEntry } from "./columns";
import { Row } from "@tanstack/react-table";
import { prefetchLogDetails } from "./prefetch";
import { RequestResponsePanel } from './RequestResponsePanel';
import { ErrorViewer } from './ErrorViewer';
import { internalUserRoles } from "../../utils/roles";
import { ConfigInfoMessage } from './ConfigInfoMessage';
import { Tooltip } from "antd";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import KeyInfoView from "../key_info_view";
import { SessionView } from './SessionView';
import { VectorStoreViewer } from './VectorStoreViewer';
import FilterComponent from "../common_components/filter";
import { FilterOption } from "../common_components/filter";
import { useLogFilterLogic } from "./log_filter_logic";

interface SpendLogsTableProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  allTeams: Team[];
}

export interface PaginatedResponse {
  data: LogEntry[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

interface PrefetchedLog {
  messages: any[];
  response: any;
}

export default function SpendLogsTable({
  accessToken,
  token,
  userRole,
  userID,
  allTeams,
}: SpendLogsTableProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [showColumnDropdown, setShowColumnDropdown] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(50);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const filtersRef = useRef<HTMLDivElement>(null);
  const quickSelectRef = useRef<HTMLDivElement>(null);

  // New state variables for Start and End Time
  const [startTime, setStartTime] = useState<string>(
    moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm")
  );
  const [endTime, setEndTime] = useState<string>(
    moment().format("YYYY-MM-DDTHH:mm")
  );

  // Add these new state variables at the top with other useState declarations
  const [isCustomDate, setIsCustomDate] = useState(false);
  const [quickSelectOpen, setQuickSelectOpen] = useState(false);
  const [tempTeamId, setTempTeamId] = useState("");
  const [tempKeyHash, setTempKeyHash] = useState("");
  const [selectedTeamId, setSelectedTeamId] = useState("");
  const [selectedKeyHash, setSelectedKeyHash] = useState("");
  const [selectedKeyInfo, setSelectedKeyInfo] = useState<KeyResponse | null>(null);
  const [selectedKeyIdInfoView, setSelectedKeyIdInfoView] = useState<string | null>(null);
  const [selectedStatus, setSelectedStatus] = useState(""); 
  const [filterByCurrentUser, setFilterByCurrentUser] = useState(
    userRole && internalUserRoles.includes(userRole)
  );

  const [expandedRequestId, setExpandedRequestId] = useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  const queryClient = useQueryClient();

  useEffect(() => {
    const fetchKeyInfo = async () => {
      if (selectedKeyIdInfoView && accessToken) {
        const keyData = await keyInfoV1Call(accessToken, selectedKeyIdInfoView);

        const keyResponse: KeyResponse = {
          ...keyData["info"],
          "token": selectedKeyIdInfoView,
          "api_key": selectedKeyIdInfoView,
        };
        setSelectedKeyInfo(keyResponse);
      }
    };
    fetchKeyInfo();
  }, [selectedKeyIdInfoView, accessToken]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setShowColumnDropdown(false);
      }
      if (
        filtersRef.current &&
        !filtersRef.current.contains(event.target as Node)
      ) {
        setShowFilters(false);
      }
      if (
        quickSelectRef.current &&
        !quickSelectRef.current.contains(event.target as Node)
      ) {
        setQuickSelectOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () =>
      document.removeEventListener("mousedown", handleClickOutside);
  }, []);


  useEffect(() => {
    if (userRole && internalUserRoles.includes(userRole)) {
      setFilterByCurrentUser(true);
    }
  }, [userRole]);

  const logs = useQuery<PaginatedResponse>({
    queryKey: [
      "logs",
      "table",
      currentPage,
      pageSize,
      startTime,
      endTime,
      selectedTeamId,
      selectedKeyHash,
      filterByCurrentUser ? userID : null,
      selectedStatus
    ],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        console.log("Missing required auth parameters");
        return {
          data: [],
          total: 0,
          page: 1,
          page_size: pageSize,
          total_pages: 0,
        };
      }

      const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTime = isCustomDate 
        ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
        : moment().utc().format("YYYY-MM-DD HH:mm:ss");

      // Get base response from API
      const response = await uiSpendLogsCall(
        accessToken,
        selectedKeyHash || undefined,
        selectedTeamId || undefined,
        undefined,
        formattedStartTime,
        formattedEndTime,
        currentPage,
        pageSize,
        filterByCurrentUser ? userID : undefined,
        selectedStatus
      );

      // Trigger prefetch for all logs
      await prefetchLogDetails(
        response.data,
        formattedStartTime,
        accessToken,
        queryClient
      );

      // Update logs with prefetched data if available
      response.data = response.data.map((log: LogEntry) => {
        const prefetchedData = queryClient.getQueryData<PrefetchedLog>(
          ["logDetails", log.request_id, formattedStartTime]
        );

        if (prefetchedData?.messages && prefetchedData?.response) {
          log.messages = prefetchedData.messages;
          log.response = prefetchedData.response;
          return log;
        }
        return log;
      });

      return response;
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID,
    refetchInterval: 5000,
    refetchIntervalInBackground: true,
  });

  const logsData = logs.data || {
    data: [],
    total: 0,
    page: 1,
    page_size: pageSize || 10,
    total_pages: 1
  };

  const {
    filters,
    filteredLogs,
    allTeams: hookAllTeams,
    allKeyAliases,
    handleFilterChange,
    handleFilterReset
  } = useLogFilterLogic({
    logs: logsData,
    accessToken,
    startTime,
    endTime,
    pageSize,
    isCustomDate,
    setCurrentPage
  })

  // Add this effect to update selectedTeamId and selectedStatus when team filter changes
  useEffect(() => {
    if (filters['Team ID']) {
      setSelectedTeamId(filters['Team ID']);
      
    } else {
      setSelectedTeamId("");
    }
    setSelectedStatus(filters['Status'] || "");
  }, [filters]);

  // Fetch logs for a session if selected
  const sessionLogs = useQuery<PaginatedResponse>({
    queryKey: ["sessionLogs", selectedSessionId],
    queryFn: async () => {
      if (!accessToken || !selectedSessionId) return { data: [], total: 0, page: 1, page_size: 50, total_pages: 1 };
      const response = await sessionSpendLogsCall(accessToken, selectedSessionId);
      // If the API returns an array, wrap it in the same shape as PaginatedResponse
      return {
        data: response.data || response || [],
        total: (response.data || response || []).length,
        page: 1,
        page_size: 1000,
        total_pages: 1,
      };
    },
    enabled: !!accessToken && !!selectedSessionId,
  });

  // Add this effect to preserve expanded state when data refreshes
  useEffect(() => {
    if (logs.data?.data && expandedRequestId) {
      // Check if the expanded request ID still exists in the new data
      const stillExists = logs.data.data.some(log => log.request_id === expandedRequestId);
      if (!stillExists) {
        // If the request ID no longer exists in the data, clear the expanded state
        setExpandedRequestId(null);
      }
    }
  }, [logs.data?.data, expandedRequestId]);

  if (!accessToken || !token || !userRole || !userID) {
    return null;
  }

  const filteredData =
    filteredLogs.data.filter((log) => {
      const matchesSearch =
        !searchTerm ||
        log.request_id.includes(searchTerm) ||
        log.model.includes(searchTerm) ||
        (log.user && log.user.includes(searchTerm));
      
      // No need for additional filtering since we're now handling this in the API call
      return matchesSearch;
      
    }).map(log => ({
      ...log,
      onKeyHashClick: (keyHash: string) => setSelectedKeyIdInfoView(keyHash),
      onSessionClick: (sessionId: string) => {
        if (sessionId) setSelectedSessionId(sessionId);
      },
    })) || [];

  // For session logs, add onKeyHashClick/onSessionClick as well
  const sessionData =
    sessionLogs.data?.data?.map(log => ({
      ...log,
      onKeyHashClick: (keyHash: string) => setSelectedKeyIdInfoView(keyHash),
      onSessionClick: (sessionId: string) => {},
    })) || [];

  // Add this function to handle manual refresh
  const handleRefresh = () => {
    logs.refetch();
  };

  // Add this function to format the time range display
  const getTimeRangeDisplay = () => {
    if (isCustomDate) {
      return `${moment(startTime).format('MMM D, h:mm A')} - ${moment(endTime).format('MMM D, h:mm A')}`;
    }
    
    const now = moment();
    const start = moment(startTime);
    const diffMinutes = now.diff(start, 'minutes');
    
    if (diffMinutes <= 15) return 'Last 15 Minutes';
    if (diffMinutes <= 60) return 'Last Hour';
    
    const diffHours = now.diff(start, 'hours');
    if (diffHours <= 4) return 'Last 4 Hours';
    if (diffHours <= 24) return 'Last 24 Hours';
    if (diffHours <= 168) return 'Last 7 Days';
    return `${start.format('MMM D')} - ${now.format('MMM D')}`;
  };

  const handleRowExpand = (requestId: string | null) => {
    setExpandedRequestId(requestId);
  };

  const logFilterOptions: FilterOption[] = [
    {
      name: 'Team ID',
      label: 'Team ID',
      isSearchable: true,
      searchFn: async (searchText: string) => {
        if (!allTeams || allTeams.length === 0) return [];
        const filtered = allTeams.filter((team: Team) =>{
          return team.team_id.toLowerCase().includes(searchText.toLowerCase()) ||
          (team.team_alias && team.team_alias.toLowerCase().includes(searchText.toLowerCase()))
      });
        return filtered.map((team: Team) => ({
          label: `${team.team_alias || team.team_id} (${team.team_id})`,
          value: team.team_id
        }));
      }
    },
    {
      name:'Status',
      label:'Status',
      isSearchable: false,
      options: [
        { label: 'Success', value: 'success' },
        { label: 'Failure', value: 'failure' }
      ]
    }
  ]

  // When a session is selected, render the SessionView component
  if (selectedSessionId && sessionLogs.data) {
    return (
      <div className="w-full p-6">
        <SessionView
          sessionId={selectedSessionId}
          logs={sessionLogs.data.data}
          onBack={() => setSelectedSessionId(null)}
        />
      </div>
    );
  }

  return (
    <div className="w-full p-6">

      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">
          {selectedSessionId ? (
            <>
              Session: <span className="font-mono">{selectedSessionId}</span>
              <button
                className="ml-4 px-3 py-1 text-sm border rounded hover:bg-gray-50"
                onClick={() => setSelectedSessionId(null)}
              >
                ← Back to All Logs
              </button>
            </>
          ) : (
            "Request Logs"
          )}
        </h1>
      </div>
      {selectedKeyInfo && selectedKeyIdInfoView && selectedKeyInfo.api_key === selectedKeyIdInfoView ? (
        <KeyInfoView keyId={selectedKeyIdInfoView} keyData={selectedKeyInfo} accessToken={accessToken} userID={userID} userRole={userRole} teams={allTeams} onClose={() => setSelectedKeyIdInfoView(null)} />
      ) : selectedSessionId ? (
        <div className="bg-white rounded-lg shadow">
          <DataTable
            columns={columns}
            data={sessionData}
            renderSubComponent={RequestViewer}
            getRowCanExpand={() => true}
            // Optionally: add session-specific row expansion state
          />
        </div>
      ) : (
        <>
        <FilterComponent options={logFilterOptions} onApplyFilters={handleFilterChange} onResetFilters={handleFilterReset} />
        <div className="bg-white rounded-lg shadow">
          <div className="border-b px-6 py-4">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0">
              <div className="flex flex-wrap items-center gap-3">
                <div className="relative w-64">
                  <input
                    type="text"
                    placeholder="Search by Request ID"
                    className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                  <svg
                    className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </div>

                <div className="flex items-center gap-2">
                  <div className="relative" ref={quickSelectRef}>
                    <button
                      onClick={() => setQuickSelectOpen(!quickSelectOpen)}
                      className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                    >
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                        />
                      </svg>
                      {getTimeRangeDisplay()}
                    </button>

                    {quickSelectOpen && (
                      <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-lg border p-2 z-50">
                        <div className="space-y-1">
                          {[
                            { label: "Last 15 Minutes", value: 15, unit: "minutes" },
                            { label: "Last Hour", value: 1, unit: "hours" },
                            { label: "Last 4 Hours", value: 4, unit: "hours" },
                            { label: "Last 24 Hours", value: 24, unit: "hours" },
                            { label: "Last 7 Days", value: 7, unit: "days" },
                          ].map((option) => (
                            <button
                              key={option.label}
                              className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md ${
                                getTimeRangeDisplay() === option.label ? 'bg-blue-50 text-blue-600' : ''
                              }`}
                              onClick={() => {
                                setEndTime(moment().format("YYYY-MM-DDTHH:mm"));
                                setStartTime(
                                  moment()
                                    .subtract(option.value, option.unit as any)
                                    .format("YYYY-MM-DDTHH:mm")
                                );
                                setQuickSelectOpen(false);
                                setIsCustomDate(false);
                              }}
                            >
                              {option.label}
                            </button>
                          ))}
                          <div className="border-t my-2" />
                          <button
                            className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md ${
                              isCustomDate ? 'bg-blue-50 text-blue-600' : ''
                            }`}
                            onClick={() => setIsCustomDate(!isCustomDate)}
                          >
                            Custom Range
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <button
                    onClick={handleRefresh}
                    className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                    title="Refresh data"
                  >
                    <svg
                      className={`w-4 h-4 ${logs.isFetching ? 'animate-spin' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                      />
                    </svg>
                    <span>Refresh</span>
                  </button>
                </div>

                {isCustomDate && (
                  <div className="flex items-center gap-2">
                    <div>
                      <input
                        type="datetime-local"
                        value={startTime}
                        onChange={(e) => {
                          setStartTime(e.target.value);
                          setCurrentPage(1);
                        }}
                        className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <span className="text-gray-500">to</span>
                    <div>
                      <input
                        type="datetime-local"
                        value={endTime}
                        onChange={(e) => {
                          setEndTime(e.target.value);
                          setCurrentPage(1);
                        }}
                        className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="flex items-center space-x-4">
                <span className="text-sm text-gray-700">
                  Showing{" "}
                  {logs.isLoading
                    ? "..."
                    : filteredLogs
                    ? (currentPage - 1) * pageSize + 1
                    : 0}{" "}
                  -{" "}
                  {logs.isLoading
                    ? "..."
                    : filteredLogs
                    ? Math.min(currentPage * pageSize, filteredLogs.total)
                    : 0}{" "}
                  of{" "}
                  {logs.isLoading
                    ? "..."
                    : filteredLogs
                    ? filteredLogs.total
                    : 0}{" "}
                  results
                </span>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-700">
                    Page {logs.isLoading ? "..." : currentPage} of{" "}
                    {logs.isLoading
                      ? "..."
                      : filteredLogs
                      ? filteredLogs.total_pages
                      : 1}
                  </span>
                  <button
                    onClick={() =>
                      setCurrentPage((p) => Math.max(1, p - 1))
                    }
                    disabled={logs.isLoading || currentPage === 1}
                    className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() =>
                      setCurrentPage((p) =>
                        Math.min(
                          filteredLogs.total_pages || 1,
                          p + 1,
                        ),
                      )
                    }
                    disabled={
                      logs.isLoading ||
                      currentPage === (filteredLogs.total_pages || 1)
                    }
                    className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          </div>
          <DataTable
            columns={columns}
            data={filteredData}
            renderSubComponent={RequestViewer}
            getRowCanExpand={() => true}
          />
        </div>
        </>
      )} 
    </div>
  );
}

export function RequestViewer({ row }: { row: Row<LogEntry> }) {
  // Helper function to clean metadata by removing specific fields
  const formatData = (input: any) => {
    if (typeof input === "string") {
      try {
        return JSON.parse(input);
      } catch {
        return input;
      }
    }
    return input;
  };

  // New helper function to get raw request
  const getRawRequest = () => {
    // First check if proxy_server_request exists in metadata
    if (row.original?.proxy_server_request) {
      return formatData(row.original.proxy_server_request);
    }
    // Fall back to messages if proxy_server_request is empty
    return formatData(row.original.messages);
  };

  // Extract error information from metadata if available
  const metadata = row.original.metadata || {};
  const hasError = metadata.status === "failure";
  const errorInfo = hasError ? metadata.error_information : null;
  
  // Check if request/response data is missing
  const hasMessages = row.original.messages && 
    (Array.isArray(row.original.messages) ? row.original.messages.length > 0 : Object.keys(row.original.messages).length > 0);
  const hasResponse = row.original.response && Object.keys(formatData(row.original.response)).length > 0;
  const missingData = !hasMessages && !hasResponse;
  
  // Format the response with error details if present
  const formattedResponse = () => {
    if (hasError && errorInfo) {
      return {
        error: {
          message: errorInfo.error_message || "An error occurred",
          type: errorInfo.error_class || "error",
          code: errorInfo.error_code  || "unknown",
          param: null
        }
      };
    }
    return formatData(row.original.response);
  };
  
  // Extract vector store request metadata if available
  const hasVectorStoreData = metadata.vector_store_request_metadata && 
    Array.isArray(metadata.vector_store_request_metadata) && 
    metadata.vector_store_request_metadata.length > 0;

  return (
    <div className="p-6 bg-gray-50 space-y-6">
      {/* Combined Info Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="p-4 border-b">
          <h3 className="text-lg font-medium">Request Details</h3>
        </div>
        <div className="grid grid-cols-2 gap-4 p-4">
          <div className="space-y-2">
            <div className="flex">
              <span className="font-medium w-1/3">Request ID:</span>
              <span className="font-mono text-sm">{row.original.request_id}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Model:</span>
              <span>{row.original.model}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Model ID:</span>
              <span>{row.original.model_id}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Call Type:</span>
              <span>{row.original.call_type}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Provider:</span>
              <span>{row.original.custom_llm_provider || "-"}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">API Base:</span>
              <Tooltip title={row.original.api_base || "-"}>
                <span className="max-w-[15ch] truncate block">{row.original.api_base || "-"}</span>
              </Tooltip>
            </div>
            {row?.original?.requester_ip_address && (
              <div className="flex">
                <span className="font-medium w-1/3">IP Address:</span>
                <span>{row?.original?.requester_ip_address}</span>
              </div>
            )}

          </div>
          <div className="space-y-2">
            <div className="flex">
              <span className="font-medium w-1/3">Tokens:</span>
              <span>{row.original.total_tokens} ({row.original.prompt_tokens}+{row.original.completion_tokens})</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Cost:</span>
              <span>${Number(row.original.spend || 0).toFixed(6)}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Cache Hit:</span>
              <span>{row.original.cache_hit}</span>
            </div>
            
            <div className="flex">
              <span className="font-medium w-1/3">Status:</span>
              <span className={`px-2 py-1 rounded-md text-xs font-medium inline-block text-center w-16 ${
                (row.original.metadata?.status || "Success").toLowerCase() !== "failure"
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {(row.original.metadata?.status || "Success").toLowerCase() !== "failure" ? "Success" : "Failure"}
              </span>
              
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Start Time:</span>
              <span>{row.original.startTime}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">End Time:</span>
              <span>{row.original.endTime}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration Info Message - Show when data is missing */}
      <ConfigInfoMessage show={missingData} />

      {/* Request/Response Panel */}
      <RequestResponsePanel
        row={row}
        hasMessages={hasMessages}
        hasResponse={hasResponse}
        hasError={hasError}
        errorInfo={errorInfo}
        getRawRequest={getRawRequest}
        formattedResponse={formattedResponse}
      />

      {/* Vector Store Request Data - Show only if present */}
      {hasVectorStoreData && (
        <VectorStoreViewer data={metadata.vector_store_request_metadata} />
      )}

      {/* Error Card - Only show for failures */}
      {hasError && errorInfo && <ErrorViewer errorInfo={errorInfo} />}

      {/* Tags Card - Only show if there are tags */}
      {row.original.request_tags && Object.keys(row.original.request_tags).length > 0 && (
        <div className="bg-white rounded-lg shadow">
          <div className="flex justify-between items-center p-4 border-b">
            <h3 className="text-lg font-medium">Request Tags</h3>
          </div>
          <div className="p-4">
            <div className="flex flex-wrap gap-2">
              {Object.entries(row.original.request_tags).map(([key, value]) => (
                <span key={key} className="px-2 py-1 bg-gray-100 rounded-full text-xs">
                  {key}: {String(value)}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Metadata Card - Only show if there's metadata */}
      {row.original.metadata && Object.keys(row.original.metadata).length > 0 && (
        <div className="bg-white rounded-lg shadow">
          <div className="flex justify-between items-center p-4 border-b">
            <h3 className="text-lg font-medium">Metadata</h3>
            <button 
              onClick={() => {
                navigator.clipboard.writeText(JSON.stringify(row.original.metadata, null, 2));
              }}
              className="p-1 hover:bg-gray-200 rounded"
              title="Copy metadata"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            </button>
          </div>
          <div className="p-4 overflow-auto max-h-64">
            <pre className="text-xs font-mono whitespace-pre-wrap break-all">
              {JSON.stringify(row.original.metadata, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}