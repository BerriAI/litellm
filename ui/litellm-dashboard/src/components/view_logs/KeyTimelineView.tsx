"use client";

import { useEffect, useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import moment from "moment";
import { concurrentTimelineCall, TimelineResponse } from "../networking";
import { istLocalToUtc, nowISTLocal, utcToISTDisplay } from "./ist_utils";

interface KeyTimelineViewProps {
  accessToken: string | null;
}

const MAX_RANGE_MINUTES = 30;

export default function KeyTimelineView({ accessToken }: KeyTimelineViewProps) {
  const [apiKey, setApiKey] = useState("");
  const [keyAlias, setKeyAlias] = useState("");

  // Time range: "Past 30 Minutes" preset or a custom IST range (max 30 minutes).
  const [isCustomDate, setIsCustomDate] = useState(false);
  const [customStart, setCustomStart] = useState<string>(nowISTLocal({ value: MAX_RANGE_MINUTES, unit: "minutes" }));
  const [customEnd, setCustomEnd] = useState<string>(nowISTLocal());

  const [submittedKey, setSubmittedKey] = useState<{ apiKey?: string; keyAlias?: string } | null>(null);
  const [validationError, setValidationError] = useState("");

  const computeRange = (): { start: string; end: string } => {
    if (isCustomDate) {
      return { start: istLocalToUtc(customStart), end: istLocalToUtc(customEnd) };
    }
    const end = moment().utc().format("YYYY-MM-DD HH:mm:ss");
    const start = moment().utc().subtract(MAX_RANGE_MINUTES, "minutes").format("YYYY-MM-DD HH:mm:ss");
    return { start, end };
  };

  // The preset is always within bounds, so only the custom range needs validation.
  const rangeError = useMemo(() => {
    if (!isCustomDate) return "";
    const startM = moment(customStart, "YYYY-MM-DDTHH:mm");
    const endM = moment(customEnd, "YYYY-MM-DDTHH:mm");
    if (!startM.isValid() || !endM.isValid()) return "";
    if (endM.isSameOrBefore(startM)) return "End time must be after start time.";
    if (endM.diff(startM, "minutes", true) > MAX_RANGE_MINUTES)
      return `Time range cannot exceed ${MAX_RANGE_MINUTES} minutes. Please select a range within ${MAX_RANGE_MINUTES} minutes.`;
    return "";
  }, [isCustomDate, customStart, customEnd]);

  const timelineQuery = useQuery<TimelineResponse>({
    queryKey: [
      "concurrencyTimeline",
      submittedKey,
      isCustomDate,
      isCustomDate ? customStart : null,
      isCustomDate ? customEnd : null,
    ],
    queryFn: async () => {
      if (!accessToken || !submittedKey) {
        return { data: [], total: 0, gcp_available: true };
      }
      const { start, end } = computeRange();
      return await concurrentTimelineCall(accessToken, {
        start_date: start,
        end_date: end,
        api_key: submittedKey.apiKey,
        key_alias: submittedKey.keyAlias,
      });
    },
    enabled: !!accessToken && !!submittedKey && !rangeError,
    placeholderData: keepPreviousData,
  });

  // Auto-fetch (debounced) as the user changes the key fields.
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
      timelineQuery.refetch();
    } else {
      setSubmittedKey(next);
    }
  };

  const result = timelineQuery.data;
  const rows = result?.data || [];
  const gcpUnavailable = !!result && (!result.gcp_available || result.gcp_success === false);

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
              value={isCustomDate ? "custom" : "preset"}
              onChange={(e) => setIsCustomDate(e.target.value === "custom")}
              className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-48"
            >
              <option value="preset">Past 30 Minutes</option>
              <option value="custom">Custom Range (IST, max 30 min)</option>
            </select>
          </div>

          <button
            onClick={handleSearch}
            disabled={timelineQuery.isFetching}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            Search
          </button>

          <button
            onClick={() => timelineQuery.refetch()}
            disabled={!submittedKey || timelineQuery.isFetching}
            className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 disabled:opacity-50"
          >
            <svg
              className={`w-4 h-4 ${timelineQuery.isFetching ? "animate-spin" : ""}`}
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

        <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded-md">
          <p className="text-sm text-amber-800">
            <strong>Note:</strong> Sampled once per minute. Redis concurrency is the last logged counter value at
            or before each minute (from GCP <span className="font-mono">[METRICS]</span> logs); Spend Logs
            concurrency counts requests active at that minute.
          </p>
        </div>
      </div>

      {gcpUnavailable && (
        <div className="mb-4 p-2 bg-amber-50 border border-amber-200 rounded-md">
          <p className="text-sm text-amber-800">
            {result?.error ||
              "GCP logging is unavailable, so Redis concurrency is shown as “—”. Spend Logs concurrency is still accurate."}
          </p>
        </div>
      )}

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Timestamp (IST)
              </th>
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
            {!submittedKey ? (
              <tr>
                <td colSpan={6} className="px-6 py-4 text-center text-gray-500">
                  Enter an API Key or Key Alias to see the per-minute timeline.
                </td>
              </tr>
            ) : timelineQuery.isError ? (
              <tr>
                <td colSpan={6} className="px-6 py-4 text-center text-red-600">
                  {(timelineQuery.error as Error)?.message || "Failed to fetch timeline."}
                </td>
              </tr>
            ) : timelineQuery.isLoading ? (
              <tr>
                <td colSpan={6} className="px-6 py-4 text-center text-gray-500">
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
            ) : rows.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-6 py-4 text-center text-gray-500">
                  No data for this key in the selected range.
                </td>
              </tr>
            ) : (
              rows.map((row, index) => (
                <tr key={`${row.timestamp}-${index}`} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                    {utcToISTDisplay(row.timestamp)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{row.key_alias}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                    {row.key_token.length > 16 ? `${row.key_token.substring(0, 16)}…` : row.key_token}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {row.spend_logs_concurrency}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        row.redis_concurrency != null && row.redis_concurrency > 0
                          ? "bg-green-100 text-green-800"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {row.redis_concurrency == null ? "—" : row.redis_concurrency}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        row.is_match == null
                          ? "bg-gray-100 text-gray-800"
                          : row.is_match
                            ? "bg-green-100 text-green-800"
                            : "bg-red-100 text-red-800"
                      }`}
                    >
                      {row.is_match == null ? "—" : row.is_match ? "Match" : "Mismatch"}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
