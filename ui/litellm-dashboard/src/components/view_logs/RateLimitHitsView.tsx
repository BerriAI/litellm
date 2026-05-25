"use client";

import { useEffect, useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import moment from "moment";
import { Tooltip } from "antd";
import { concurrentRateLimitHitsCall, RateLimitHitsResponse } from "../networking";
import { QUICK_SELECT_OPTIONS } from "./constants";
import { istLocalToUtc, nowISTLocal, utcToISTDisplay } from "./ist_utils";

interface RateLimitHitsViewProps {
  accessToken: string | null;
}

const PAGE_SIZE = 10;

export default function RateLimitHitsView({ accessToken }: RateLimitHitsViewProps) {
  const [apiKey, setApiKey] = useState("");
  const [keyAlias, setKeyAlias] = useState("");

  // Time range (live). Quick-select uses a relative window; custom uses IST inputs.
  const [selectedInterval, setSelectedInterval] = useState<{ value: number; unit: string }>({
    value: 24,
    unit: "hours",
  });
  const [isCustomDate, setIsCustomDate] = useState(false);
  const [customStart, setCustomStart] = useState<string>(nowISTLocal({ value: 24, unit: "hours" }));
  const [customEnd, setCustomEnd] = useState<string>(nowISTLocal());

  // The validated key snapshot that actually drives the query. Updated only on Search.
  const [submittedKey, setSubmittedKey] = useState<{ apiKey?: string; keyAlias?: string } | null>(null);
  const [validationError, setValidationError] = useState("");
  const [currentPage, setCurrentPage] = useState(1);

  // Reset to first page whenever the query inputs change.
  useEffect(() => {
    setCurrentPage(1);
  }, [submittedKey, selectedInterval, isCustomDate, customStart, customEnd]);

  const computeRange = (): { start: string; end: string } => {
    if (isCustomDate) {
      return { start: istLocalToUtc(customStart), end: istLocalToUtc(customEnd) };
    }
    const end = moment().utc().format("YYYY-MM-DD HH:mm:ss");
    const start = moment()
      .utc()
      .subtract(selectedInterval.value, selectedInterval.unit as moment.unitOfTime.DurationConstructor)
      .format("YYYY-MM-DD HH:mm:ss");
    return { start, end };
  };

  // Validate the custom range (quick-select options max out at 7 days, so they're
  // always within bounds). Both inputs share the same tz, so the diff is offset-safe.
  const rangeError = useMemo(() => {
    if (!isCustomDate) return "";
    const startM = moment(customStart, "YYYY-MM-DDTHH:mm");
    const endM = moment(customEnd, "YYYY-MM-DDTHH:mm");
    if (!startM.isValid() || !endM.isValid()) return "";
    if (endM.isSameOrBefore(startM)) return "End time must be after start time.";
    if (endM.diff(startM, "days", true) > 10) return "Time range cannot exceed 10 days.";
    return "";
  }, [isCustomDate, customStart, customEnd]);

  const hitsQuery = useQuery<RateLimitHitsResponse>({
    queryKey: [
      "rateLimitHits",
      submittedKey,
      selectedInterval,
      isCustomDate,
      isCustomDate ? customStart : null,
      isCustomDate ? customEnd : null,
      currentPage,
    ],
    queryFn: async () => {
      if (!accessToken || !submittedKey) {
        return { data: [], total: 0, page: 1, page_size: PAGE_SIZE, total_pages: 0 };
      }
      const { start, end } = computeRange();
      return await concurrentRateLimitHitsCall(accessToken, {
        start_date: start,
        end_date: end,
        api_key: submittedKey.apiKey,
        key_alias: submittedKey.keyAlias,
        page: currentPage,
        page_size: PAGE_SIZE,
      });
    },
    enabled: !!accessToken && !!submittedKey && !rangeError,
    placeholderData: keepPreviousData,
  });

  // Auto-fetch as the user changes the key fields (debounced so we don't fire a
  // request on every keystroke). Exactly one of API Key / Key Alias must be set.
  useEffect(() => {
    const trimmedKey = apiKey.trim();
    const trimmedAlias = keyAlias.trim();

    if (trimmedKey && trimmedAlias) {
      setValidationError("Please provide only one of API Key or Key Alias, not both.");
      setSubmittedKey(null);
      return;
    }
    if (!trimmedKey && !trimmedAlias) {
      setValidationError("");
      setSubmittedKey(null);
      return;
    }

    setValidationError("");
    const handle = setTimeout(() => {
      setSubmittedKey(trimmedKey ? { apiKey: trimmedKey } : { keyAlias: trimmedAlias });
    }, 400);
    return () => clearTimeout(handle);
  }, [apiKey, keyAlias]);

  // Explicit search: applies the current key immediately (bypassing the debounce).
  // If the key is unchanged, the query key won't change, so force a refetch.
  const handleSearch = () => {
    const trimmedKey = apiKey.trim();
    const trimmedAlias = keyAlias.trim();

    if (trimmedKey && trimmedAlias) {
      setValidationError("Please provide only one of API Key or Key Alias, not both.");
      setSubmittedKey(null);
      return;
    }
    if (!trimmedKey && !trimmedAlias) {
      setValidationError("Please provide either an API Key or a Key Alias.");
      setSubmittedKey(null);
      return;
    }

    setValidationError("");
    const next = trimmedKey ? { apiKey: trimmedKey } : { keyAlias: trimmedAlias };
    const isSameKey =
      !!submittedKey &&
      submittedKey.apiKey === next.apiKey &&
      submittedKey.keyAlias === next.keyAlias;
    if (isSameKey) {
      hitsQuery.refetch();
    } else {
      setSubmittedKey(next);
    }
  };

  const data = hitsQuery.data?.data || [];
  const total = hitsQuery.data?.total || 0;
  const totalPages = hitsQuery.data?.total_pages || 0;

  const rangeLabel = useMemo(() => {
    if (isCustomDate) return "Custom Range (IST)";
    return QUICK_SELECT_OPTIONS.find(
      (o) => o.value === selectedInterval.value && o.unit === selectedInterval.unit
    )?.label;
  }, [isCustomDate, selectedInterval]);

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full">
      {/* Inputs */}
      <div className="bg-white rounded-lg shadow p-4 mb-4">
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">API Key</label>
            <input
              type="text"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Masked key as shown in UI, e.g. sk-...gQxg"
              className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-56"
            />
          </div>

          <div className="flex items-center text-sm text-gray-400 pb-2">or</div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">Key Alias</label>
            <input
              type="text"
              value={keyAlias}
              onChange={(e) => setKeyAlias(e.target.value)}
              placeholder="e.g., production-key"
              className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-56"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">Time Range</label>
            <select
              value={isCustomDate ? "custom" : `${selectedInterval.value}-${selectedInterval.unit}`}
              onChange={(e) => {
                if (e.target.value === "custom") {
                  setIsCustomDate(true);
                  return;
                }
                setIsCustomDate(false);
                const opt = QUICK_SELECT_OPTIONS.find((o) => `${o.value}-${o.unit}` === e.target.value);
                if (opt) setSelectedInterval({ value: opt.value, unit: opt.unit });
              }}
              className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-44"
            >
              {QUICK_SELECT_OPTIONS.map((o) => (
                <option key={`${o.value}-${o.unit}`} value={`${o.value}-${o.unit}`}>
                  {o.label}
                </option>
              ))}
              <option value="custom">Custom Range (IST)</option>
            </select>
          </div>

          <button
            onClick={handleSearch}
            disabled={hitsQuery.isFetching}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            Search
          </button>

          <button
            onClick={() => hitsQuery.refetch()}
            disabled={!submittedKey || hitsQuery.isFetching}
            className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 disabled:opacity-50"
          >
            <svg
              className={`w-4 h-4 ${hitsQuery.isFetching ? "animate-spin" : ""}`}
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

        {isCustomDate && (
          <div className="flex items-center gap-2 mt-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-gray-500">Start (IST)</label>
              <input
                type="datetime-local"
                value={customStart}
                onChange={(e) => setCustomStart(e.target.value)}
                className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <span className="text-gray-500 pt-5">to</span>
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-gray-500">End (IST)</label>
              <input
                type="datetime-local"
                value={customEnd}
                onChange={(e) => setCustomEnd(e.target.value)}
                className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>
        )}

        {(validationError || rangeError) && (
          <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-700">{validationError || rangeError}</p>
          </div>
        )}

        {submittedKey && !validationError && !rangeError && (
          <div className="mt-2 text-sm text-gray-500">
            Showing max_parallel_requests rate limit hits for{" "}
            <span className="font-mono">
              {submittedKey.apiKey
                ? `API Key ${submittedKey.apiKey.length > 24 ? submittedKey.apiKey.substring(0, 24) + "…" : submittedKey.apiKey}`
                : `Key Alias "${submittedKey.keyAlias}"`}
            </span>{" "}
            over <span className="font-medium">{rangeLabel}</span>
            {total > 0 && <span className="ml-2">({total} hits)</span>}
          </div>
        )}
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                #
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Timestamp (IST)
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Model
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Limit Details
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {!submittedKey ? (
              <tr>
                <td colSpan={4} className="px-6 py-4 text-center text-gray-500">
                  Enter an API Key or Key Alias to see results.
                </td>
              </tr>
            ) : hitsQuery.isError ? (
              <tr>
                <td colSpan={4} className="px-6 py-4 text-center text-red-600">
                  {(hitsQuery.error as Error)?.message || "Failed to fetch rate limit hits."}
                </td>
              </tr>
            ) : hitsQuery.isLoading ? (
              <tr>
                <td colSpan={4} className="px-6 py-4 text-center text-gray-500">
                  <div className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
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
            ) : data.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-6 py-4 text-center text-gray-500">
                  No max_parallel_requests rate limit hits found for this key in the selected range.
                </td>
              </tr>
            ) : (
              data.map((row, index) => (
                <tr key={`${row.request_id}-${index}`} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {(currentPage - 1) * PAGE_SIZE + index + 1}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                    {utcToISTDisplay(row.timestamp)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{row.model || "-"}</td>
                  <td className="px-6 py-4 text-sm text-gray-500 max-w-md">
                    <Tooltip title={row.error_message}>
                      <span className="block truncate">{row.error_message || "-"}</span>
                    </Tooltip>
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
                disabled={currentPage === 1 || hitsQuery.isFetching}
                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages || hitsQuery.isFetching}
                className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded-md">
        <p className="text-sm text-amber-800">
          <strong>Note:</strong> Timestamps come from SpendLogs failure entries (status 429,
          <span className="font-mono"> Limit type: max_parallel_requests</span>) and are shown in IST.
        </p>
      </div>
    </div>
  );
}
