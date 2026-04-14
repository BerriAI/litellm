"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import moment from "moment";
import { concurrentRequestLogsPaginatedCall } from "../networking";

interface ConcurrentRequestLogsProps {
  accessToken: string | null;
}

interface ConcurrentRequestData {
  key_alias: string;
  key_token: string;
  spend_logs_concurrency: number | string;  // number when available, "—" when cannot query
  redis_concurrency: number;
  is_match: boolean;  // True if Redis and SpendLogs concurrency values match
}

type MatchFilter = "all" | "matching" | "mismatching";

const PAGE_SIZE = 10;

// Convert UTC to IST timestamp string for datetime-local input
// Returns "now - 10 minutes" to ensure SpendLogs entries exist ( SpendLogs are written after request completes)
const getNowIST = (): string => {
  // Get current UTC moment, subtract 10 minutes, then convert to IST (+05:30)
  return moment().utc().subtract(10, 'minutes').add(5, 'hours').add(30, 'minutes').format("YYYY-MM-DDTHH:mm:ss.SSS");
};

// Get minimum allowed timestamp (10 minutes ago from now)
const getMinTimestampIST = (): string => {
  // Minimum allowed is 10 minutes ago from current time
  return moment().utc().subtract(10, 'minutes').add(5, 'hours').add(30, 'minutes').format("YYYY-MM-DDTHH:mm:ss.SSS");
};

// Convert IST input to ISO timestamp (UTC) for API
const istToISO = (istTimestamp: string): string => {
  // Parse as IST and convert to UTC
  return moment.utc(istTimestamp).subtract(5, 'hours').subtract(30, 'minutes').toISOString();
};

// Format IST timestamp for display
// Parse as UTC to be consistent with istToISO (we treat IST values as UTC for calculation)
const formatIST = (istTimestamp: string): string => {
  return moment.utc(istTimestamp).format("YYYY-MM-DD HH:mm:ss.SSS [IST]");
};

export default function ConcurrentRequestLogs({
  accessToken,
}: ConcurrentRequestLogsProps) {
  const [targetTimestamp, setTargetTimestamp] = useState<string>(getNowIST());
  const [apiKey, setApiKey] = useState<string>("");
  const [keyAlias, setKeyAlias] = useState<string>("");
  const [matchFilter, setMatchFilter] = useState<MatchFilter>("all");
  const [currentPage, setCurrentPage] = useState(1);

  // Fetch concurrent request logs (GCP Logs + SpendLogs combined)
  // Server-side filtering for matchStatus - filtered server-side for proper pagination
  const logsData = useQuery<{
    data: ConcurrentRequestData[];
    total: number;
  }>({
    queryKey: ["concurrentRequestLogs", targetTimestamp, currentPage, apiKey, keyAlias, matchFilter],
    queryFn: async () => {
      if (!accessToken) return { data: [], total: 0 };
      const isoTimestamp = istToISO(targetTimestamp);
      return await concurrentRequestLogsPaginatedCall(
        accessToken,
        isoTimestamp,
        currentPage,
        PAGE_SIZE,
        apiKey || undefined,
        keyAlias || undefined,
        matchFilter !== "all" ? matchFilter : undefined
      );
    },
    enabled: !!accessToken,
  });

  // Use data directly from server (match filtering is now server-side)
  const convertedData = logsData.data?.data || [];
  const totalItems = logsData.data?.total || 0;

  const handleRefresh = () => {
    logsData.refetch();
  };

  const handleResetToNow = () => {
    setTargetTimestamp(getNowIST());
    setCurrentPage(1);
  };

  const formattedTimestamp = formatIST(targetTimestamp);
  // Total from server already accounts for filters
  const totalPages = Math.ceil(totalItems / PAGE_SIZE);

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Concurrent Request Logs</h1>
      </div>

      {/* Input Fields */}
      <div className="bg-white rounded-lg shadow p-4 mb-4">
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">
              Timestamp (IST, millisecond precision)
            </label>
            <input
              type="datetime-local"
              step="0.001"
              value={targetTimestamp}
              max={getMinTimestampIST()}
              onChange={(e) => {
                setTargetTimestamp(e.target.value);
                setCurrentPage(1);
              }}
              className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">
              API Key (optional)
            </label>
            <input
              type="text"
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                setCurrentPage(1);
              }}
              placeholder="e.g., sk-...gQxg"
              className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-48"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">
              Key Alias (optional)
            </label>
            <input
              type="text"
              value={keyAlias}
              onChange={(e) => {
                setKeyAlias(e.target.value);
                setCurrentPage(1);
              }}
              placeholder="e.g., production-key"
              className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-48"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">
              Match Status
            </label>
            <select
              value={matchFilter}
              onChange={(e) => {
                setMatchFilter(e.target.value as MatchFilter);
                setCurrentPage(1);
              }}
              className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-40"
            >
              <option value="all">All</option>
              <option value="matching">Matching</option>
              <option value="mismatching">Mismatching</option>
            </select>
          </div>

          <button
            onClick={handleResetToNow}
            className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50"
          >
            Reset to 10 mins ago
          </button>

          <button
            onClick={handleRefresh}
            disabled={logsData.isLoading}
            className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 disabled:opacity-50"
          >
            <svg
              className={`w-4 h-4 ${logsData.isLoading ? "animate-spin" : ""}`}
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
            Refresh
          </button>
        </div>

        <div className="mt-2 text-sm text-gray-500">
          Querying at: <span className="font-mono">{formattedTimestamp}</span>
          {totalItems > 0 && (
            <span className="ml-4">
              ({totalItems} keys with active requests
              {matchFilter !== "all" && ` - ${matchFilter} only`})
            </span>
          )}
        </div>

        <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded-md">
          <p className="text-sm text-amber-800">
            <strong>Note:</strong> SpendLogs entries are created after request completion.
            Timestamps within the last 10 minutes may show incomplete data.
          </p>
        </div>
      </div>

      {/* Data Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Key Alias
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Key Token
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Spend Logs Concurrency
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Redis Concurrency
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Match Status
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {logsData.isLoading ? (
              <tr>
                <td colSpan={5} className="px-6 py-4 text-center text-gray-500">
                  <div className="flex items-center justify-center gap-2">
                    <svg
                      className="animate-spin h-5 w-5 text-gray-400"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Loading...
                  </div>
                </td>
              </tr>
            ) : convertedData.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-4 text-center text-gray-500">
                  No active concurrent requests found at this timestamp
                </td>
              </tr>
            ) : (
              convertedData.map((row, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {row.key_alias}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                    {row.key_token.substring(0, 16)}...
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      typeof row.spend_logs_concurrency === 'number'
                        ? "bg-blue-100 text-blue-800"
                        : "bg-gray-100 text-gray-800"
                    }`}>
                      {row.spend_logs_concurrency}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        row.redis_concurrency > 0
                          ? "bg-green-100 text-green-800"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {row.redis_concurrency}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        row.is_match
                          ? "bg-green-100 text-green-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {row.is_match ? "Match" : "Mismatch"}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="px-6 py-4 border-t flex items-center justify-between">
            <span className="text-sm text-gray-700">
              Page {currentPage} of {totalPages}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1 || logsData.isLoading}
                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages || logsData.isLoading}
                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 text-sm text-gray-500">
        <p>
          <strong>Spend Logs Concurrency:</strong> Count of requests in SpendLogs
          table that were active at the target timestamp (started within 60 min
          before and not yet ended).
        </p>
        <p className="mt-1">
          <strong>Redis Concurrency:</strong> Value from Redis parallel request
          counters logged to GCP Cloud Logging within the last 5 seconds before
          the target timestamp.
        </p>
      </div>
    </div>
  );
}
