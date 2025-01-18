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
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowColumnDropdown(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const logs = useQuery({
    queryKey: ["logs", "table"],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        console.log(
          "got None values for one of accessToken, token, userRole, userID",
        );
        return;
      }

      // Get logs for last 24 hours using ISO string and proper date formatting
      const endTime = moment().format("YYYY-MM-DD HH:mm:ss");
      const startTime = moment()
        .subtract(24, "hours")
        .format("YYYY-MM-DD HH:mm:ss");

      const data = await uiSpendLogsCall(
        accessToken,
        token,
        userRole,
        userID,
        startTime,
        endTime,
      );

      return data;
    },
  });

  if (!accessToken || !token || !userRole || !userID) {
    console.log(
      "got None values for one of accessToken, token, userRole, userID",
    );
    return null;
  }

  const filteredData = logs.data?.filter((log) => {
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
    <div className="px-4 md:px-8 py-8 w-full">
      <h1 className="text-xl font-semibold mb-4">Traces</h1>
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-4 border-b">
          <div className="flex items-center gap-3">
            <div className="relative flex-1 max-w-md">
              <input
                type="text"
                placeholder="Search by id, name, user id"
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
                        onClick={() => setShowColumnDropdown(!showColumnDropdown)}
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
                                selectedColumn === option ? 'bg-blue-50 text-blue-600' : ''
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
                                <svg className="h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
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
            <select className="px-3 py-2 border rounded-md bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 flex items-center gap-2">
              <option>24 hours</option>
              <option>7 days</option>
              <option>30 days</option>
            </select>
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
          {JSON.stringify(formatData(row.original.request_id), null, 2)}
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
