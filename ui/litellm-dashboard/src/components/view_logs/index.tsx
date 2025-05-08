import moment from "moment";
import React, { useState, useRef, useEffect, useMemo } from "react";
import { useQueryClient, useQuery } from "@tanstack/react-query";
import { ColumnDef, Row } from "@tanstack/react-table";

import { uiSpendLogsCall, keyInfoV1Call, sessionSpendLogsCall } from "../networking";
import { DataTable } from "./table";
import { LogEntry } from "./columns";
import { getCountryFromIP } from "./ip_lookup";
import { CountryCell } from "./country_cell";
import { TimeCell } from "./time_cell";
import { prefetchLogDetails } from "./prefetch";
import { RequestResponsePanel } from "./columns";
import { ErrorViewer } from './ErrorViewer';
import { internalUserRoles } from "../../utils/roles";
import { ConfigInfoMessage } from './ConfigInfoMessage';
import { Tooltip, Button as AntButton } from "antd";
import { Button } from "@tremor/react";
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
  const [pageSize] = useState(50);
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
  const [selectedKeyInfo, setSelectedKeyInfo] = useState<KeyResponse | null>(null);
  const [selectedKeyIdInfoView, setSelectedKeyIdInfoView] = useState<string | null>(null);
  const [expandedRequestId, setExpandedRequestId] = useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  const queryClient = useQueryClient();

  const {
    filters,
    filteredLogs,
    allTeams: hookAllTeams,
    allOrganizations,
    allUsers,
    allKeyAliases,
    allModels,
    handleFilterChange,
    handleFilterReset,
    isLoading,
    pagination,
    setCurrentPage,
  } = useLogFilterLogic({
    accessToken,
    startTime,
    endTime,
    pageSize,
  });

  const sessionLogs = useQuery<{data: LogEntry[]}>({
    queryKey: ["sessionLogs", selectedSessionId],
    queryFn: async () => {
      if (!accessToken || !selectedSessionId) return { data: [] };
      const response = await sessionSpendLogsCall(accessToken, selectedSessionId);
      return {
        data: response.data || response || [],
      };
    },
    enabled: !!accessToken && !!selectedSessionId,
  });

  useEffect(() => {
    if (filteredLogs && expandedRequestId) {
      const stillExists = filteredLogs.some((log: LogEntry) => log.request_id === expandedRequestId);
      if (!stillExists) {
        setExpandedRequestId(null);
      }
    }
  }, [filteredLogs, expandedRequestId]);

  useEffect(() => {
    const fetchKeyInfo = async () => {
      if (selectedKeyIdInfoView && accessToken) {
        const keyData = await keyInfoV1Call(accessToken, selectedKeyIdInfoView);
        console.log("keyData", keyData);

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

  if (!accessToken || !userID) {
    console.log(
      "Missing accessToken or userID",
    );
    return null;
  }

  const processedLogs = filteredLogs.filter((log: LogEntry) => {
    const matchesSearch =
      !searchTerm ||
      log.request_id.includes(searchTerm) ||
      log.model.includes(searchTerm) ||
      (log.user && log.user.includes(searchTerm));
    return matchesSearch;
  }).map((log: LogEntry) => ({
    ...log,
    onKeyHashClick: (keyHash: string) => setSelectedKeyIdInfoView(keyHash),
    onSessionClick: (sessionId: string) => {
      if (sessionId) setSelectedSessionId(sessionId);
    },
  }));

  const sessionData =
    sessionLogs.data?.data?.map((log: LogEntry) => ({
      ...log,
      onKeyHashClick: (keyHash: string) => setSelectedKeyIdInfoView(keyHash),
      onSessionClick: (sessionId: string) => {},
    })) || [];

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['logs', 'table', accessToken, startTime, endTime, filters] });
  };

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
        if (!hookAllTeams || hookAllTeams.length === 0) return [];
        const filtered = hookAllTeams.filter((team: Team) =>{
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
      name: 'Status',
      label: 'Status',
      isSearchable: false,
      options: [
        { label: 'All', value: '' },
        { label: 'Success', value: 'success' },
        { label: 'Failure', value: 'failure' },
      ],
    },
    {
      name: 'Model',
      label: 'Model',
      isSearchable: true, 
      searchFn: async (searchText: string) => {
        console.log("allModels", allModels);
        const filteredModels = allModels.filter(model => 
          model.toLowerCase().includes(searchText.toLowerCase())
        );
        return filteredModels.map(model => ({
          label: model,
          value: model
        }));
      }
    }
    // {
    //   name: 'Key Alias',
    //   label: 'Key Alias',
    //   isSearchable: true,
    //   searchFn: async (searchText: string) => {
    //     const filteredKeyAliases = allKeyAliases.filter(key => {
    //       console.log("key", searchText);
    //       return key.toLowerCase().includes(searchText.toLowerCase())
    //     });

    //     return filteredKeyAliases.map((key) => {
    //       return {
    //         label: key,
    //         value: key
    //       }
    //     });
    //   }
    // },
    // {
    //   name: 'Key Hash',
    //   label: 'Key Hash',
    //   isSearchable: false, 
    // },
    // {
    //   name: 'Request ID',
    //   label: 'Request ID',
    //   isSearchable: false,
    // },
    // {
    //   name: 'Model',
    //   label: 'Model',
    //   isSearchable: false, 
    // },
    // {
    //     name: 'User',
    //     label: 'User',
    //     isSearchable: true,
    //     searchFn: async (searchText: string) => {
    //         if (!allUsers || allUsers.length === 0) return [];
    //         const filtered = allUsers.filter((user: UserInfo) => 
    //           (user.user_id && user.user_id.toLowerCase().includes(searchText.toLowerCase())) ||
    //           (user.user_email && user.user_email.toLowerCase().includes(searchText.toLowerCase()))
    //         );
    //         return filtered.map((user: UserInfo) => ({
    //           label: `${user.user_email || user.user_id} (${user.user_id})`,
    //           value: user.user_id
    //         }));
    //     }
    // },
  ];

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

  const columns: ColumnDef<LogEntry>[] = [
    {
        id: "expander",
        header: () => null,
        cell: ({ row }) => {
          const ExpanderCell = () => {
            const [localExpanded, setLocalExpanded] = React.useState(row.getIsExpanded());
            const toggleHandler = React.useCallback(() => {
              setLocalExpanded((prev) => !prev);
              row.getToggleExpandedHandler()();
            }, [row]);

            return row.getCanExpand() ? (
              <button
                onClick={toggleHandler}
                style={{ cursor: "pointer" }}
                aria-label={localExpanded ? "Collapse row" : "Expand row"}
                className="w-6 h-6 flex items-center justify-center focus:outline-none"
              >
                 <svg
                    className={`w-4 h-4 transform transition-transform duration-75 ${
                        localExpanded ? 'rotate-90' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                    >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 5l7 7-7 7"
                    />
                    </svg>
              </button>
            ) : (
              <span className="w-6 h-6 flex items-center justify-center">●</span>
            );
          };
          return <ExpanderCell />;
        },
    },
    {
        header: "Time",
        accessorKey: "startTime",
        cell: (info: any) => <TimeCell utcTime={info.getValue()} />,
    },
    {
      header: "Status",
      accessorFn: (row) => row.metadata?.status || "Success",
      cell: (info: any) => {
        const status = info.getValue() as string;
        const isSuccess = status.toLowerCase() !== "failure";
        return (
          <span className={`px-2 py-1 rounded-md text-xs font-medium inline-block text-center w-16 ${
            isSuccess 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {isSuccess ? "Success" : "Failure"}
          </span>
        );
      },
    },
    {
        header: "Session ID",
        accessorKey: "session_id",
        cell: (info: any) => {
          const value = String(info.getValue() || "");
          const onSessionClick = info.row.original.onSessionClick;
          return value ? (
            <Tooltip title={value}>
            <Button 
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal text-xs max-w-[15ch] truncate block"
              onClick={() => onSessionClick?.(value)}
            >
              {value}
            </Button>
          </Tooltip>
          ) : "-";
        },
    },
    {
        header: "Request ID",
        accessorKey: "request_id",
        cell: (info: any) => (
          <Tooltip title={String(info.getValue() || "")}>
            <span className="font-mono text-xs max-w-[15ch] truncate block">
              {String(info.getValue() || "")}
            </span>
          </Tooltip>
        ),
    },
    {
        header: "Cost",
        accessorKey: "spend",
        cell: (info: any) => (
          <span>${Number(info.getValue() || 0).toFixed(6)}</span>
        ),
    },
    {
      header: "Country",
      accessorKey: "requester_ip_address",
      cell: (info: any) => <CountryCell ipAddress={info.getValue()} />,
    },
    {
      header: "Team Name",
      accessorKey: "team_id",
      cell: ({ row }) => {
        const teamId = row.original.team_id;
        if (!teamId) return "-";
        const team = hookAllTeams.find(t => t.team_id === teamId);
        const teamDisplayName = team?.team_alias || teamId;
        const truncatedName = teamDisplayName.length > 20 
          ? `${teamDisplayName.slice(0, 20)}...` 
          : teamDisplayName;

        return (
          <Tooltip title={teamDisplayName}>
            <span className="max-w-[15ch] truncate block">{truncatedName}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Key Hash",
      accessorFn: (row) => row.metadata?.user_api_key || "-",
      cell: (info: any) => {
        const value = String(info.getValue());
        const onKeyHashClick = info.row.original.onKeyHashClick;
        return value !== "-" ? (
          <Tooltip title={value}>
            <span 
              className="font-mono max-w-[15ch] truncate block cursor-pointer hover:text-blue-600"
              onClick={() => onKeyHashClick?.(value)}
            >
              {value}
            </span>
          </Tooltip>
        ) : "-";
      },
    },
    {
      header: "Key Name",
      accessorFn: (row) => row.metadata?.user_api_key_alias || "-",
      cell: (info: any) => (
        <Tooltip title={String(info.getValue())}>
          <span className="max-w-[15ch] truncate block">{String(info.getValue())}</span>
        </Tooltip>
      ),
    },
    {
        header: "User",
        accessorKey: "user",
        cell: (info: any) => (
            <Tooltip title={String(info.getValue() || "-")}>
                <span className="max-w-[15ch] truncate block">{String(info.getValue() || "-")}</span>
            </Tooltip>
        ),
    },
    {
        header: "End User",
        accessorKey: "end_user",
        cell: (info: any) => (
            <Tooltip title={String(info.getValue() || "-")}>
                <span className="max-w-[15ch] truncate block">{String(info.getValue() || "-")}</span>
            </Tooltip>
        ),
    },
    {
        header: "Model",
        accessorKey: "model",
        cell: (info: any) => (
            <Tooltip title={String(info.getValue() || "-")}>
                <span className="max-w-[15ch] truncate block">{String(info.getValue() || "-")}</span>
            </Tooltip>
        ),
    },
    {
        header: "Provider",
        accessorKey: "custom_llm_provider",
        cell: (info: any) => (
            <Tooltip title={String(info.getValue() || "-")}>
                <span className="max-w-[15ch] truncate block">{String(info.getValue() || "-")}</span>
            </Tooltip>
        ),
    },
    {
        header: "Cache",
        accessorKey: "cache_hit",
        cell: (info: any) => info.getValue() || "-",
    },
    {
        header: "Tokens",
        accessorKey: "total_tokens",
        cell: (info: any) => {
            const total = info.row.original.total_tokens || 0;
            const prompt = info.row.original.prompt_tokens || 0;
            const completion = info.row.original.completion_tokens || 0;
            return (
                <Tooltip title={`Prompt: ${prompt}, Completion: ${completion}`}>
                    <span>{total}</span>
                </Tooltip>
            );
        },
    },
  ];

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
    <div className="w-full p-6 overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">
          Request Logs
        </h1>
      </div>
      {selectedKeyInfo && selectedKeyIdInfoView && selectedKeyInfo.api_key === selectedKeyIdInfoView ? (
        <KeyInfoView keyId={selectedKeyIdInfoView} keyData={selectedKeyInfo} accessToken={accessToken} userID={userID} userRole={userRole} teams={hookAllTeams} onClose={() => setSelectedKeyIdInfoView(null)} />
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
                    placeholder="Search by Request ID, Model, User"
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
                    className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`}
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

                <div className="flex items-center space-x-4">
                  <span className="text-sm text-gray-700">
                    Showing{" "}
                    {isLoading
                      ? "..."
                      : pagination.totalCount > 0
                      ? (pagination.currentPage - 1) * pagination.pageSize + 1
                      : 0}{" "}
                    -{" "}
                    {isLoading
                      ? "..."
                      : pagination.totalCount > 0
                      ? Math.min(pagination.currentPage * pagination.pageSize, pagination.totalCount)
                      : 0}{" "}
                    of{" "}
                    {isLoading
                      ? "..."
                      : pagination.totalCount ?? 0}{" "}
                    results
                  </span>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-700">
                      Page {isLoading ? "..." : pagination.currentPage} of{" "}
                      {isLoading
                        ? "..."
                        : pagination.totalPages ?? 1}
                    </span>
                    <button
                      onClick={() =>
                        setCurrentPage((p) => Math.max(1, p - 1))
                      }
                      disabled={isLoading || pagination.currentPage === 1}
                      className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    <button
                      onClick={() =>
                        setCurrentPage((p) =>
                          Math.min(
                            pagination.totalPages || 1,
                            p + 1,
                          ),
                        )
                      }
                      disabled={
                        isLoading ||
                        pagination.currentPage === (pagination.totalPages || 1)
                      }
                      className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                </div>
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
          </div>
          <DataTable
            columns={columns}
            data={processedLogs}
            renderSubComponent={RequestViewer}
            getRowCanExpand={() => true}
            isLoading={isLoading}
          />
        </div>
        </>
      )}
    </div>
  );
}

export function RequestViewer({ row }: { row: Row<LogEntry> }) {
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

  const getRawRequest = () => {
    if (row.original?.proxy_server_request) {
      return formatData(row.original.proxy_server_request);
    }
    return formatData(row.original.messages);
  };

  const metadata = row.original.metadata || {};
  const hasError = metadata.status === "failure";
  const errorInfo = hasError ? metadata.error_information : null;
  
  const hasMessages = row.original.messages && 
    (Array.isArray(row.original.messages) ? row.original.messages.length > 0 : Object.keys(row.original.messages).length > 0);
  const hasResponse = row.original.response && Object.keys(formatData(row.original.response)).length > 0;
  const missingData = !hasMessages && !hasResponse;
  
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
  
  const hasVectorStoreData = metadata.vector_store_request_metadata && 
    Array.isArray(metadata.vector_store_request_metadata) && 
    metadata.vector_store_request_metadata.length > 0;

  return (
    <div className="p-6 bg-gray-50 space-y-6">
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
              <span className="font-medium w-1/3">Provider:</span>
              <span>{row.original.custom_llm_provider || "-"}</span>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">API Base:</span>
              <Tooltip title={row.original.api_base || "-"}>
                <span className="max-w-[15ch] truncate block">{row.original.api_base || "-"}</span>
              </Tooltip>
            </div>
            <div className="flex">
              <span className="font-medium w-1/3">Start Time:</span>
              <span>{row.original.startTime}</span>
            </div>

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
            {row?.original?.requester_ip_address && (
              <div className="flex">
                <span className="font-medium w-1/3">IP Address:</span>
                <span>{row?.original?.requester_ip_address}</span>
              </div>
            )}
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
              <span className="font-medium w-1/3">End Time:</span>
              <span>{row.original.endTime}</span>
            </div>
          </div>
        </div>
      </div>

      <ConfigInfoMessage show={missingData} />

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded-lg shadow">
          <div className="flex justify-between items-center p-4 border-b">
            <h3 className="text-lg font-medium">Request</h3>
            <button 
              onClick={() => navigator.clipboard.writeText(JSON.stringify(getRawRequest(), null, 2))}
              className="p-1 hover:bg-gray-200 rounded"
              title="Copy request"
              disabled={!hasMessages}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            </button>
          </div>
          <div className="p-4 overflow-auto max-h-96">
            <pre className="text-xs font-mono whitespace-pre-wrap break-all">{JSON.stringify(getRawRequest(), null, 2)}</pre>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow">
          <div className="flex justify-between items-center p-4 border-b">
            <h3 className="text-lg font-medium">
              Response
              {hasError && (
                <span className="ml-2 text-sm text-red-600">
                  • HTTP code {errorInfo?.error_code || 400}
                </span>
              )}
            </h3>
            <button 
              onClick={() => navigator.clipboard.writeText(JSON.stringify(formattedResponse(), null, 2))}
              className="p-1 hover:bg-gray-200 rounded"
              title="Copy response"
              disabled={!hasResponse}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            </button>
          </div>
          <div className="p-4 overflow-auto max-h-96 bg-gray-50">
            {hasResponse ? (
              <pre className="text-xs font-mono whitespace-pre-wrap break-all">{JSON.stringify(formattedResponse(), null, 2)}</pre>
            ) : (
              <div className="text-gray-500 text-sm italic text-center py-4">Response data not available</div>
            )}
          </div>
        </div>
      </div>

      {hasVectorStoreData && (
        <VectorStoreViewer data={metadata.vector_store_request_metadata} />
      )}

      {hasError && errorInfo && <ErrorViewer errorInfo={errorInfo} />}

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