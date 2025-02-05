import moment from "moment";
import { useQuery } from "@tanstack/react-query";
import { useState, useRef, useEffect } from "react";

import { uiSpendLogsCall } from "../networking";
import { DataTable } from "./table";
import { columns, LogEntry } from "./columns";
import { Row } from "@tanstack/react-table";

interface SpendLogsTableProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
}

interface PaginatedResponse {
  data: LogEntry[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export default function SpendLogsTable({
  accessToken,
  token,
  userRole,
  userID,
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
  const [selectedFilter, setSelectedFilter] = useState("Team ID");

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

      // Convert times to UTC before formatting
      const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTime = isCustomDate 
        ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
        : moment().utc().format("YYYY-MM-DD HH:mm:ss");

      return await uiSpendLogsCall(
        accessToken,
        selectedKeyHash || undefined,
        selectedTeamId || undefined,
        undefined,
        formattedStartTime,
        formattedEndTime,
        currentPage,
        pageSize
      );
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID,
    refetchInterval: 5000,
    refetchIntervalInBackground: true,
  });

  if (!accessToken || !token || !userRole || !userID) {
    console.log(
      "got None values for one of accessToken, token, userRole, userID",
    );
    return null;
  }

  const filteredData =
    logs.data?.data?.filter((log) => {
      const matchesSearch =
        !searchTerm ||
        log.request_id.includes(searchTerm) ||
        log.model.includes(searchTerm) ||
        (log.user && log.user.includes(searchTerm));
      
      // No need for additional filtering since we're now handling this in the API call
      return matchesSearch;
    }) || [];

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

  return (
    <div className="w-full p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Request Logs</h1>
      </div>
      
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
              <div className="relative" ref={filtersRef}>
                <button
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                  onClick={() => setShowFilters(!showFilters)}
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
                      d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                    />
                  </svg>
                  Filter
                </button>

                {showFilters && (
                  <div className="absolute left-0 mt-2 w-[500px] bg-white rounded-lg shadow-lg border p-4 z-50">
                    <div className="flex flex-col gap-4">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">Where</span>
                        <div className="relative">
                          <button
                            onClick={() => setShowColumnDropdown(!showColumnDropdown)}
                            className="px-3 py-1.5 border rounded-md bg-white text-sm min-w-[160px] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-left flex justify-between items-center"
                          >
                            {selectedFilter}
                            <svg
                              className="h-4 w-4 text-gray-500"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M19 9l-7 7-7-7"
                              />
                            </svg>
                          </button>
                          {showColumnDropdown && (
                            <div className="absolute left-0 mt-1 w-[160px] bg-white border rounded-md shadow-lg z-50">
                              {["Team ID", "Key Hash"].map((option) => (
                                <button
                                  key={option}
                                  className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 flex items-center gap-2 ${
                                    selectedFilter === option
                                      ? "bg-blue-50 text-blue-600"
                                      : ""
                                  }`}
                                  onClick={() => {
                                    setSelectedFilter(option);
                                    setShowColumnDropdown(false);
                                    if (option === "Team ID") {
                                      setTempKeyHash("");
                                    } else {
                                      setTempTeamId("");
                                    }
                                  }}
                                >
                                  {selectedFilter === option && (
                                    <svg
                                      className="h-4 w-4 text-blue-600"
                                      fill="none"
                                      viewBox="0 0 24 24"
                                      stroke="currentColor"
                                    >
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M5 13l4 4L19 7"
                                      />
                                    </svg>
                                  )}
                                  {option}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                        <input
                          type="text"
                          placeholder="Enter value..."
                          className="px-3 py-1.5 border rounded-md text-sm flex-1 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          value={selectedFilter === "Team ID" ? tempTeamId : tempKeyHash}
                          onChange={(e) => {
                            if (selectedFilter === "Team ID") {
                              setTempTeamId(e.target.value);
                            } else {
                              setTempKeyHash(e.target.value);
                            }
                          }}
                        />
                        <button
                          className="p-1 hover:bg-gray-100 rounded-md"
                          onClick={() => {
                            setTempTeamId("");
                            setTempKeyHash("");
                          }}
                        >
                          <span className="text-gray-500">×</span>
                        </button>
                      </div>
                      
                      <div className="flex justify-end gap-2">
                        <button
                          className="px-3 py-1.5 text-sm border rounded-md hover:bg-gray-50"
                          onClick={() => {
                            setTempTeamId("");
                            setTempKeyHash("");
                            setShowFilters(false);
                          }}
                        >
                          Cancel
                        </button>
                        <button
                          className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
                          onClick={() => {
                            setSelectedTeamId(tempTeamId);
                            setSelectedKeyHash(tempKeyHash);
                            setCurrentPage(1); // Reset to first page when applying new filters
                            setShowFilters(false);
                          }}
                        >
                          Apply Filters
                        </button>
                      </div>
                    </div>
                  </div>
                )}
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
                  : logs.data
                  ? (currentPage - 1) * pageSize + 1
                  : 0}{" "}
                -{" "}
                {logs.isLoading
                  ? "..."
                  : logs.data
                  ? Math.min(currentPage * pageSize, logs.data.total)
                  : 0}{" "}
                of{" "}
                {logs.isLoading
                  ? "..."
                  : logs.data
                  ? logs.data.total
                  : 0}{" "}
                results
              </span>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-700">
                  Page {logs.isLoading ? "..." : currentPage} of{" "}
                  {logs.isLoading
                    ? "..."
                    : logs.data
                    ? logs.data.total_pages
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
                        logs.data?.total_pages || 1,
                        p + 1,
                      ),
                    )
                  }
                  disabled={
                    logs.isLoading ||
                    currentPage === (logs.data?.total_pages || 1)
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
    </div>
  );
}

function RequestViewer({ row }: { row: Row<LogEntry> }) {
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

  return (
    <div className="p-6 bg-gray-50 space-y-6">
      {/* Combined Info Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="p-4 border-b">
          <h3 className="text-lg font-medium ">Request Details</h3>
        </div>
        <div className="space-y-2 p-4 ">
          <div className="flex">
            <span className="font-medium w-1/3">Request ID:</span>
            <span>{row.original.request_id}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Api Key:</span>
            <span>{row.original.api_key}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Team ID:</span>
            <span>{row.original.team_id}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Model:</span>
            <span>{row.original.model}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Api Base:</span>
            <span>{row.original.api_base}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Call Type:</span>
            <span>{row.original.call_type}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Spend:</span>
            <span>{row.original.spend}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Total Tokens:</span>
            <span>{row.original.total_tokens}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Prompt Tokens:</span>
            <span>{row.original.prompt_tokens}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Completion Tokens:</span>
            <span>{row.original.completion_tokens}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Start Time:</span>
            <span>{row.original.startTime}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">End Time:</span>
            <span>{row.original.endTime}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Cache Hit:</span>
            <span>{row.original.cache_hit}</span>
          </div>
          <div className="flex">
            <span className="font-medium w-1/3">Cache Key:</span>
            <span>{row.original.cache_key}</span>
          </div>
          {row?.original?.requester_ip_address && (
            <div className="flex">
              <span className="font-medium w-1/3">Request IP Address:</span>
              <span>{row?.original?.requester_ip_address}</span>
            </div>
          )}
        </div>
      </div>

      {/* Request Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Request Tags</h3>
        </div>
        <pre className="p-4  text-wrap overflow-auto text-sm">
          {JSON.stringify(formatData(row.original.request_tags), null, 2)}
        </pre>
      </div>

      {/* Request Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Request</h3>
          {/* <div>
            <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">
              Expand
            </button>
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
              JSON
            </button>
          </div> */}
        </div>
        <pre className="p-4  text-wrap overflow-auto text-sm">
          {JSON.stringify(formatData(row.original.messages), null, 2)}
        </pre>
      </div>

      {/* Response Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Response</h3>
          <div>
            {/* <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">
              Expand
            </button>
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
              JSON
            </button> */}
          </div>
        </div>
        <pre className="p-4 text-wrap overflow-auto text-sm">
          {JSON.stringify(formatData(row.original.response), null, 2)}
        </pre>
      </div>

      {/* Metadata Card */}
      {row.original.metadata &&
        Object.keys(row.original.metadata).length > 0 && (
          <div className="bg-white rounded-lg shadow">
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="text-lg font-medium">Metadata</h3>
              {/* <div>
                <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
                  JSON
                </button>
              </div> */}
            </div>
            <pre className="p-4 text-wrap  overflow-auto text-sm ">
              {JSON.stringify(row.original.metadata, null, 2)}
            </pre>
          </div>
        )}
    </div>
  );
}
