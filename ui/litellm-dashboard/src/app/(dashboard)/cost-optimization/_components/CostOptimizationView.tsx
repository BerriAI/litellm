"use client";

import React, { useMemo, useState } from "react";
import { PiggyBank } from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { AreaChart, DonutChart } from "@/components/shared/charts";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getCostOptimizationUsageLogs,
  type OptimizedRequestLog,
  type OptimizedRequestLogsResponse,
  userDailyActivityCall,
} from "@/components/networking";
import { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { all_admin_roles } from "@/utils/roles";
import { usePaginatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/usePaginatedDailyActivity";

interface CostOptimizationViewProps {
  accessToken: string | null;
  userId: string | null;
  userRole: string;
}

type DateRange = { from?: Date; to?: Date };

const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

const usd = (value: number): string => {
  const decimals = value > 0 && value < 1 ? 4 : 2;
  return `$${formatNumberWithCommas(value, decimals)}`;
};

const shortDate = (iso: string): string =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });

const compressionOf = (m: SpendMetrics): number => m.compression_savings_spend ?? 0;
const cachingOf = (m: SpendMetrics): number => m.prompt_caching_savings_spend ?? 0;
const savedTokensOf = (m: SpendMetrics): number => m.compression_saved_tokens ?? 0;

const SummaryCard = ({ label, value, hint }: { label: string; value: string; hint?: string }) => (
  <Card>
    <CardHeader>
      <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
    </CardHeader>
    <CardContent>
      <p className="text-2xl font-semibold text-foreground">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </CardContent>
  </Card>
);

const optimizationTypeLabel = (type: OptimizedRequestLog["optimization_type"]): string => {
  if (type === "both") return "Both";
  if (type === "caching") return "Caching";
  return "Compression";
};

const optimizationTypeClass = (type: OptimizedRequestLog["optimization_type"]): string => {
  if (type === "both") return "bg-purple-50 text-purple-700 border-purple-200";
  if (type === "caching") return "bg-green-50 text-green-700 border-green-200";
  return "bg-blue-50 text-blue-700 border-blue-200";
};

const optimizedLogsDescription = (
  loading: boolean,
  fetching: boolean,
  error: unknown,
  data: OptimizedRequestLogsResponse | undefined,
): string => {
  if (loading || fetching) return "Loading...";
  if (error) return "Failed to load optimized requests";
  if (data?.total) return `Showing ${data.logs.length} of ${data.total} requests`;
  return "No optimized requests for this period";
};

const OptimizedRequestsTable = ({
  data,
  logsPage,
  onPrevious,
  onNext,
}: {
  data: OptimizedRequestLogsResponse;
  logsPage: number;
  onPrevious: () => void;
  onNext: () => void;
}) => (
  <>
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="px-3 py-3 font-medium">Request ID</th>
            <th className="px-3 py-3 font-medium">Timestamp</th>
            <th className="px-3 py-3 font-medium">Model</th>
            <th className="px-3 py-3 text-right font-medium">Tokens</th>
            <th className="px-3 py-3 font-medium">Type</th>
            <th className="px-3 py-3 text-right font-medium">Original Cost</th>
            <th className="px-3 py-3 text-right font-medium">Optimized Cost</th>
            <th className="px-3 py-3 text-right font-medium">Savings</th>
          </tr>
        </thead>
        <tbody>
          {data.logs.map((log) => (
            <tr key={log.request_id} className="border-b last:border-0">
              <td className="max-w-40 truncate px-3 py-3 font-mono text-xs" title={log.request_id}>
                {log.request_id}
              </td>
              <td className="whitespace-nowrap px-3 py-3 text-muted-foreground">
                {new Date(log.timestamp).toLocaleString()}
              </td>
              <td className="max-w-48 truncate px-3 py-3" title={log.model}>
                {log.model}
              </td>
              <td className="px-3 py-3 text-right">{formatNumberWithCommas(log.total_tokens)}</td>
              <td className="px-3 py-3">
                <span
                  className={`inline-flex rounded border px-2 py-0.5 text-xs font-medium ${optimizationTypeClass(log.optimization_type)}`}
                >
                  {optimizationTypeLabel(log.optimization_type)}
                </span>
              </td>
              <td className="px-3 py-3 text-right text-muted-foreground line-through">{usd(log.original_cost)}</td>
              <td className="px-3 py-3 text-right">{usd(log.spend)}</td>
              <td className="px-3 py-3 text-right font-medium text-emerald-600">{usd(log.savings)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    {data.total_pages > 1 && (
      <div className="mt-4 flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          Page {data.page} of {data.total_pages}
        </p>
        <div className="flex gap-2">
          <button
            type="button"
            className="rounded border px-3 py-1 text-sm disabled:cursor-not-allowed disabled:opacity-50"
            disabled={logsPage === 1}
            onClick={onPrevious}
          >
            Previous
          </button>
          <button
            type="button"
            className="rounded border px-3 py-1 text-sm disabled:cursor-not-allowed disabled:opacity-50"
            disabled={logsPage >= data.total_pages}
            onClick={onNext}
          >
            Next
          </button>
        </div>
      </div>
    )}
  </>
);

const CostOptimizationView: React.FC<CostOptimizationViewProps> = ({ accessToken, userId, userRole }) => {
  const initialFrom = useMemo(() => new Date(new Date().getTime() - THIRTY_DAYS_MS), []);
  const initialTo = useMemo(() => new Date(), []);
  const [dateValue, setDateValue] = useState<DateRange>({ from: initialFrom, to: initialTo });

  const startTime = dateValue.from ?? null;
  const endTime = dateValue.to ?? null;
  const isAdmin = all_admin_roles.includes(userRole);
  const effectiveUserId = isAdmin ? null : userId;

  const { data, loading, isFetchingMore } = usePaginatedDailyActivity({
    fetchFn: userDailyActivityCall,
    args: [accessToken, startTime, endTime, effectiveUserId],
    enabled: !!accessToken && !!startTime && !!endTime,
  });

  const results = data.results as DailyData[];

  const compressionTotal = useMemo(() => results.reduce((sum, d) => sum + compressionOf(d.metrics), 0), [results]);
  const cachingTotal = useMemo(() => results.reduce((sum, d) => sum + cachingOf(d.metrics), 0), [results]);
  const savedTokensTotal = useMemo(() => results.reduce((sum, d) => sum + savedTokensOf(d.metrics), 0), [results]);
  const totalSaved = compressionTotal + cachingTotal;
  const [logsPage, setLogsPage] = useState(1);
  const startDate = startTime ? startTime.toISOString().slice(0, 10) : "";
  const endDate = endTime ? endTime.toISOString().slice(0, 10) : "";
  const {
    data: optimizedLogsData,
    isLoading: optimizedLogsLoading,
    isFetching: optimizedLogsFetching,
    error: optimizedLogsError,
  } = useQuery({
    queryKey: ["cost-optimization-usage-logs", startDate, endDate, logsPage],
    queryFn: () =>
      getCostOptimizationUsageLogs({
        accessToken: accessToken!,
        startDate,
        endDate,
        page: logsPage,
        pageSize: 50,
      }),
    enabled: !!accessToken && !!startDate && !!endDate,
  });
  const logsDescription = optimizedLogsDescription(
    optimizedLogsLoading,
    optimizedLogsFetching,
    optimizedLogsError,
    optimizedLogsData,
  );

  const overTime = useMemo(
    () =>
      results.map((d) => ({
        date: shortDate(d.date),
        Compression: compressionOf(d.metrics),
        "Prompt caching": cachingOf(d.metrics),
      })),
    [results],
  );

  const byDriver = useMemo(
    () =>
      [
        { driver: "Compression", usd: compressionTotal },
        { driver: "Prompt caching", usd: cachingTotal },
      ].filter((d) => d.usd > 0),
    [compressionTotal, cachingTotal],
  );

  return (
    <div className="w-full space-y-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <PiggyBank className="size-6 text-emerald-600" strokeWidth={1.75} />
            <h1 className="text-xl font-semibold text-foreground">Cost Optimization</h1>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Money saved by prompt compression and prompt caching across your requests
          </p>
        </div>
        <AdvancedDatePicker
          value={dateValue}
          onValueChange={(v) => {
            setDateValue(v);
            setLogsPage(1);
          }}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <SummaryCard
          label="Total saved"
          value={usd(totalSaved)}
          hint={loading || isFetchingMore ? "Loading..." : "Compression + prompt caching"}
        />
        <SummaryCard
          label="Compression savings"
          value={usd(compressionTotal)}
          hint={`${formatNumberWithCommas(savedTokensTotal)} tokens compressed`}
        />
        <SummaryCard label="Prompt caching savings" value={usd(cachingTotal)} hint="Cache read discount" />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Savings over time</CardTitle>
          </CardHeader>
          <CardContent>
            <AreaChart
              data={overTime}
              index="date"
              categories={["Compression", "Prompt caching"]}
              colors={["emerald", "blue"]}
              valueFormatter={usd}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Savings by driver</CardTitle>
          </CardHeader>
          <CardContent>
            <DonutChart
              className="h-80"
              data={byDriver}
              index="driver"
              category="usd"
              colors={["emerald", "blue"]}
              valueFormatter={usd}
              showLabel
              label={usd(totalSaved)}
            />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Optimized Requests</CardTitle>
          <p className="text-sm text-muted-foreground">{logsDescription}</p>
        </CardHeader>
        <CardContent>
          {optimizedLogsLoading && (
            <div className="py-10 text-center text-sm text-muted-foreground">Loading optimized requests...</div>
          )}
          {!optimizedLogsLoading && optimizedLogsError && (
            <div className="py-10 text-center text-sm text-red-600">Failed to load optimized requests.</div>
          )}
          {!optimizedLogsLoading && !optimizedLogsError && optimizedLogsData?.logs.length ? (
            <OptimizedRequestsTable
              data={optimizedLogsData}
              logsPage={logsPage}
              onPrevious={() => setLogsPage((page) => Math.max(1, page - 1))}
              onNext={() => setLogsPage((page) => Math.min(optimizedLogsData.total_pages, page + 1))}
            />
          ) : null}
          {!optimizedLogsLoading && !optimizedLogsError && !optimizedLogsData?.logs.length && (
            <div className="py-10 text-center text-sm text-muted-foreground">No optimized requests to display.</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default CostOptimizationView;
