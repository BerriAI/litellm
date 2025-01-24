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
  const [keyNameFilter, setKeyNameFilter] = useState("");
  const [teamNameFilter, setTeamNameFilter] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [showColumnDropdown, setShowColumnDropdown] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState("Key Name");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(50);
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setShowColumnDropdown(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () =>
      document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const logs = useQuery<PaginatedResponse>({
    queryKey: ["logs", "table", currentPage, pageSize, startTime, endTime],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        console.log(
          "got None values for one of accessToken, token, userRole, userID",
        );
        return {
          data: [],
          total: 0,
          page: 1,
          page_size: pageSize,
          total_pages: 0,
        };
      }

      const formattedStartTime = moment(startTime).format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTime = moment(endTime).format("YYYY-MM-DD HH:mm:ss");

      const data = await uiSpendLogsCall(
        accessToken,
        token,
        userRole,
        userID,
        formattedStartTime,
        formattedEndTime,
        currentPage,
        pageSize,
      );

      return data;
    },
    // Refetch when startTime or endTime changes
    enabled: !!accessToken && !!token && !!userRole && !!userID,
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
      const matchesKeyName =
        !keyNameFilter ||
        (log.metadata?.user_api_key_alias &&
          log.metadata.user_api_key_alias
            .toLowerCase()
            .includes(keyNameFilter.toLowerCase()));
      const matchesTeamName =
        !teamNameFilter ||
        (log.metadata?.user_api_key_team_alias &&
          log.metadata.user_api_key_team_alias
            .toLowerCase()
            .includes(teamNameFilter.toLowerCase()));
      return matchesSearch && matchesKeyName && matchesTeamName;
    }) || [];

  return (
    <div className="w-full">
      <h1 className="text-xl font-semibold mb-4">Traces</h1>
      <div className="bg-white rounded-lg shadow overflow-hidden">
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
              <div className="relative" ref={dropdownRef}>
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
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">Where</span>
                      <div className="relative">
                        <button
                          onClick={() =>
                            setShowColumnDropdown(!showColumnDropdown)
                          }
                          className="px-3 py-1.5 border rounded-md bg-white text-sm min-w-[160px] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-left flex justify-between items-center"
                        >
                          {selectedColumn}
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
                            {["Key Name", "Team Name"].map((option) => (
                              <button
                                key={option}
                                className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 flex items-center gap-2 ${
                                  selectedColumn === option
                                    ? "bg-blue-50 text-blue-600"
                                    : ""
                                }`}
                                onClick={() => {
                                  setSelectedColumn(option);
                                  setShowColumnDropdown(false);
                                  if (option === "Key Name") {
                                    setTeamNameFilter("");
                                  } else {
                                    setKeyNameFilter("");
                                  }
                                }}
                              >
                                {selectedColumn === option && (
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
                        value={keyNameFilter || teamNameFilter}
                        onChange={(e) => {
                          if (selectedColumn === "Key Name") {
                            setKeyNameFilter(e.target.value);
                          } else {
                            setTeamNameFilter(e.target.value);
                          }
                        }}
                      />
                      <button
                        className="p-1 hover:bg-gray-100 rounded-md"
                        onClick={() => {
                          setKeyNameFilter("");
                          setTeamNameFilter("");
                          setShowFilters(false);
                        }}
                      >
                        <span className="text-gray-500">×</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>

              <div className="relative">
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
                  Time Range
                </button>

                {quickSelectOpen && (
                  <div className="absolute left-0 mt-2 w-64 bg-white rounded-lg shadow-lg border p-2 z-50">
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
                          className="w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md"
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
                        className="w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md"
                        onClick={() => setIsCustomDate(!isCustomDate)}
                      >
                        Custom Range
                      </button>
                    </div>
                  </div>
                )}
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
      {/* Request Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Request</h3>
          <div>
            <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">
              Expand
            </button>
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
              JSON
            </button>
          </div>
        </div>
        <pre className="p-4 overflow-auto text-sm">
          {JSON.stringify(formatData(row.original.messages), null, 2)}
        </pre>
      </div>

      {/* Response Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Response</h3>
          <div>
            <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">
              Expand
            </button>
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
              JSON
            </button>
          </div>
        </div>
        <pre className="p-4 overflow-auto text-sm">
          {JSON.stringify(formatData(row.original.response), null, 2)}
        </pre>
      </div>

      {/* Metadata Card */}
      {row.original.metadata &&
        Object.keys(row.original.metadata).length > 0 && (
          <div className="bg-white rounded-lg shadow">
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="text-lg font-medium">Metadata</h3>
              <div>
                <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">
                  JSON
                </button>
              </div>
            </div>
            <pre className="p-4 overflow-auto text-sm">
              {JSON.stringify(row.original.metadata, null, 2)}
            </pre>
          </div>
        )}
    </div>
  );
}
