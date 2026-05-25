"use client";

import { useEffect, useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import moment from "moment";
import { concurrentOperationCountsCall, OperationCountsResponse } from "../networking";
import { QUICK_SELECT_OPTIONS } from "./constants";
import { istLocalToUtc, nowISTLocal } from "./ist_utils";

interface CounterOpsViewProps {
  accessToken: string | null;
}

export default function CounterOpsView({ accessToken }: CounterOpsViewProps) {
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

  // The validated key snapshot that actually drives the query.
  const [submittedKey, setSubmittedKey] = useState<{ apiKey?: string; keyAlias?: string } | null>(null);
  const [validationError, setValidationError] = useState("");

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
    if (endM.diff(startM, "days", true) > 10) return "Time range cannot exceed 10 days. Please select a range within 10 days.";
    return "";
  }, [isCustomDate, customStart, customEnd]);

  const countsQuery = useQuery<OperationCountsResponse>({
    queryKey: [
      "operationCounts",
      submittedKey,
      selectedInterval,
      isCustomDate,
      isCustomDate ? customStart : null,
      isCustomDate ? customEnd : null,
    ],
    queryFn: async () => {
      if (!accessToken || !submittedKey) {
        return {
          increment_count: 0,
          decrement_count: 0,
          difference: 0,
          truncated: false,
          gcp_available: true,
        };
      }
      const { start, end } = computeRange();
      return await concurrentOperationCountsCall(accessToken, {
        start_date: start,
        end_date: end,
        api_key: submittedKey.apiKey,
        key_alias: submittedKey.keyAlias,
      });
    },
    enabled: !!accessToken && !!submittedKey && !rangeError,
    placeholderData: keepPreviousData,
  });

  // Auto-fetch (debounced) as the user changes the key fields. Exactly one of
  // API Key / Key Alias must be set.
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
      countsQuery.refetch();
    } else {
      setSubmittedKey(next);
    }
  };

  const result = countsQuery.data;
  const rangeLabel = useMemo(() => {
    if (isCustomDate) return "Custom Range (IST)";
    return QUICK_SELECT_OPTIONS.find(
      (o) => o.value === selectedInterval.value && o.unit === selectedInterval.unit
    )?.label;
  }, [isCustomDate, selectedInterval]);

  if (!accessToken) {
    return null;
  }

  const hasResult = !!submittedKey && !!result && !countsQuery.isError && result.gcp_available && !result.error;
  const difference = result?.difference ?? 0;

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
            disabled={countsQuery.isFetching}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            Search
          </button>

          <button
            onClick={() => countsQuery.refetch()}
            disabled={!submittedKey || countsQuery.isFetching}
            className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 disabled:opacity-50"
          >
            <svg
              className={`w-4 h-4 ${countsQuery.isFetching ? "animate-spin" : ""}`}
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
            Showing increment / decrement counts for{" "}
            <span className="font-mono">
              {submittedKey.apiKey
                ? `API Key ${submittedKey.apiKey.length > 24 ? submittedKey.apiKey.substring(0, 24) + "…" : submittedKey.apiKey}`
                : `Key Alias "${submittedKey.keyAlias}"`}
            </span>{" "}
            over <span className="font-medium">{rangeLabel}</span>
          </div>
        )}
      </div>

      {/* Results */}
      <div className="bg-white rounded-lg shadow p-6">
        {!submittedKey ? (
          <div className="text-center text-gray-500 py-6">Enter an API Key or Key Alias to see counts.</div>
        ) : countsQuery.isError ? (
          <div className="text-center text-red-600 py-6">
            {(countsQuery.error as Error)?.message || "Failed to fetch operation counts."}
          </div>
        ) : countsQuery.isLoading ? (
          <div className="flex items-center justify-center gap-2 text-gray-500 py-6">
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
        ) : result && (!result.gcp_available || result.error) ? (
          <div className="text-center text-amber-700 py-6">
            {result.error || "GCP logging is not available on the server."}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="rounded-lg border border-green-200 bg-green-50 p-4">
                <div className="text-sm font-medium text-green-700">Increment Logs</div>
                <div className="mt-1 text-3xl font-bold text-green-800">
                  {(result?.increment_count ?? 0).toLocaleString()}
                </div>
                <div className="mt-1 text-xs text-green-600">operation=increment</div>
              </div>

              <div className="rounded-lg border border-orange-200 bg-orange-50 p-4">
                <div className="text-sm font-medium text-orange-700">Decrement Logs</div>
                <div className="mt-1 text-3xl font-bold text-orange-800">
                  {(result?.decrement_count ?? 0).toLocaleString()}
                </div>
                <div className="mt-1 text-xs text-orange-600">operation=decrement</div>
              </div>

              <div
                className={`rounded-lg border p-4 ${
                  difference === 0 ? "border-blue-200 bg-blue-50" : "border-red-200 bg-red-50"
                }`}
              >
                <div className={`text-sm font-medium ${difference === 0 ? "text-blue-700" : "text-red-700"}`}>
                  Difference (inc − dec)
                </div>
                <div className={`mt-1 text-3xl font-bold ${difference === 0 ? "text-blue-800" : "text-red-800"}`}>
                  {difference > 0 ? `+${difference.toLocaleString()}` : difference.toLocaleString()}
                </div>
                <div className={`mt-1 text-xs ${difference === 0 ? "text-blue-600" : "text-red-600"}`}>
                  {difference === 0 ? "Balanced" : "Increments and decrements differ"}
                </div>
              </div>
            </div>

            {hasResult && result?.truncated && (
              <div className="mt-4 p-2 bg-amber-50 border border-amber-200 rounded-md">
                <p className="text-sm text-amber-800">
                  <strong>Note:</strong> Hit the maximum number of log entries scanned, so these counts are a
                  lower bound. Narrow the time range for exact counts.
                </p>
              </div>
            )}
          </>
        )}
      </div>

      <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded-md">
        <p className="text-sm text-amber-800">
          <strong>Note:</strong> Counts come from the parallel_requests <span className="font-mono">[METRICS]</span>{" "}
          log lines in GCP Cloud Logging. An API Key is resolved to its token first (these logs carry token /
          key_alias, not the masked key).
        </p>
      </div>
    </div>
  );
}
