import { DataTable } from "./table";
import { RequestViewer } from ".";
import moment from "moment";
import { useRef, useState } from "react";
import { getTimeRangeDisplay } from "./logs_utils";
import { useQuery } from "@tanstack/react-query";
import { PaginatedResponse } from ".";
import { uiAuditLogsCall } from "../networking";
import { AuditLogEntry, auditLogColumns } from "./columns";
import { Text } from "@tremor/react";

interface AuditLogsProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  isActive: boolean;
  premiumUser: boolean;
}

export interface PaginatedAuditLogResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  audit_logs: AuditLogEntry[];
}

export default function AuditLogs({
  userID,
  userRole,
  token,
  accessToken,
  isActive,
  premiumUser,
}: AuditLogsProps) {
  const [startTime, setStartTime] = useState<string>(
    moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm")
  );
  const [endTime, setEndTime] = useState<string>(
    moment().format("YYYY-MM-DDTHH:mm")
  );

  const quickSelectRef = useRef<HTMLDivElement>(null);
  const [quickSelectOpen, setQuickSelectOpen] = useState(false);
  const [isCustomDate, setIsCustomDate] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(50);

  const logs = useQuery<PaginatedAuditLogResponse>({
    queryKey: [
      "logs",
      "table",
      currentPage,
      pageSize,
      startTime,
      endTime
    ],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        return {
          total: 0,
          page: 1,
          page_size: pageSize,
          total_pages: 0,
          audit_logs: [],
        };
      }

      const formattedStartTime = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTime = isCustomDate 
        ? moment(endTime).utc().format("YYYY-MM-DD HH:mm:ss")
        : moment().utc().format("YYYY-MM-DD HH:mm:ss");

      // Get base response from API
      const response = await uiAuditLogsCall(
        accessToken,
        formattedStartTime,
        formattedEndTime,
        currentPage,
        pageSize,
      );

      return response;
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && isActive,
    refetchInterval: 5000,
    refetchIntervalInBackground: true,
  });

  // Add this function to handle manual refresh
  const handleRefresh = () => {
    logs.refetch();
  };
  
  if (!premiumUser) {
    return (
      <div>
        <Text>This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key <a href="https://litellm.ai/pricing" target="_blank" rel="noopener noreferrer">here</a>.</Text>
      </div>
    );
  }

  return (
    <>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">
          Audit Logs
        </h1>
      </div>
      <div className="bg-white rounded-lg shadow">
        <div className="border-b px-6 py-4">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0">
            <div className="flex flex-wrap items-center gap-3">
              

              <div className="flex items-center gap-2">
                <div className="relative" ref={quickSelectRef}>
                  {/* <button
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
                    {getTimeRangeDisplay(isCustomDate, startTime, endTime)}
                  </button> */}

                  {/* {quickSelectOpen && (
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
                              getTimeRangeDisplay(isCustomDate, startTime, endTime) === option.label ? 'bg-blue-50 text-blue-600' : ''
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
                  )} */}
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
                  : logs
                  ? (currentPage - 1) * pageSize + 1
                  : 0}{" "}
                -{" "}
                {logs.isLoading
                  ? "..."
                  : logs
                  ? Math.min(currentPage * pageSize, logs.data?.total ?? 0)
                  : 0}{" "}
                of{" "}
                {logs.isLoading
                  ? "..."
                  : logs
                  ? logs.data?.total
                  : 0}{" "}
                results
              </span>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-700">
                  Page {logs.isLoading ? "..." : currentPage} of{" "}
                  {logs.isLoading
                    ? "..."
                    : logs
                    ? logs.data?.total_pages
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
          columns={auditLogColumns}
          data={logs.data?.audit_logs ?? []}
          renderSubComponent={() => {return <></>}}
          getRowCanExpand={() => true}
        />
      </div>
    </>
  );
}
