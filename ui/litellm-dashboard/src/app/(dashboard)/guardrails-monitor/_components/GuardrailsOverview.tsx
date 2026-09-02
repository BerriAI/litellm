import { useQuery } from "@tanstack/react-query";
import type { ColumnDef, OnChangeFn, SortingState } from "@tanstack/react-table";
import { Download, HeartPulse, Settings, TrendingUp, TriangleAlert } from "lucide-react";
import React, { useMemo, useState } from "react";
import { DataTable, DataTableSortHeader } from "@/components/shared/DataTable";
import { getGuardrailsUsageOverview } from "@/components/networking";
import { type PerformanceRow } from "@/components/GuardrailsMonitor/mockData";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/shared/PageHeader";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { EvaluationSettingsModal } from "./EvaluationSettingsModal";
import { MetricCard } from "@/components/GuardrailsMonitor/MetricCard";
import { ScoreChart } from "./ScoreChart";

interface GuardrailsOverviewProps {
  accessToken?: string | null;
  startDate: string;
  endDate: string;
  onSelectGuardrail: (id: string) => void;
  dateRangeControl?: React.ReactNode;
}

type SortKey = "failRate" | "requestsEvaluated" | "avgLatency" | "falsePositiveRate" | "falseNegativeRate";

const providerColors: Record<string, string> = {
  Bedrock: "bg-warning/15 text-warning border-warning/20",
  "Google Cloud": "bg-info/15 text-info border-info/20",
  LiteLLM:
    "bg-indigo-100 text-indigo-700 border-indigo-200 dark:bg-indigo-950 dark:text-indigo-300 dark:border-indigo-800",
  Custom: "bg-muted text-muted-foreground border-border",
};

function computeMetricsFromRows(data: PerformanceRow[]) {
  const totalRequests = data.reduce((sum, r) => sum + r.requestsEvaluated, 0);
  const totalBlocked = data.reduce((sum, r) => sum + Math.round((r.requestsEvaluated * r.failRate) / 100), 0);
  const passRate = totalRequests > 0 ? ((1 - totalBlocked / totalRequests) * 100).toFixed(1) : "0";
  const withLat = data.filter((r) => r.avgLatency != null);
  const avgLatency =
    withLat.length > 0 ? Math.round(withLat.reduce((sum, r) => sum + (r.avgLatency ?? 0), 0) / withLat.length) : 0;
  return { totalRequests, totalBlocked, passRate, avgLatency, count: data.length };
}

export function GuardrailsOverview({
  accessToken = null,
  startDate,
  endDate,
  onSelectGuardrail,
  dateRangeControl,
}: GuardrailsOverviewProps) {
  const [sortBy, setSortBy] = useState<SortKey>("failRate");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [evaluationModalOpen, setEvaluationModalOpen] = useState(false);

  const {
    data: guardrailsData,
    isLoading: guardrailsLoading,
    error: guardrailsError,
  } = useQuery({
    queryKey: ["guardrails-usage-overview", startDate, endDate],
    queryFn: () => getGuardrailsUsageOverview(accessToken!, startDate, endDate),
    enabled: !!accessToken,
  });

  const activeData: PerformanceRow[] = guardrailsData?.rows ?? [];
  const metrics = useMemo(() => {
    if (guardrailsData) {
      return {
        totalRequests: guardrailsData.totalRequests ?? 0,
        totalBlocked: guardrailsData.totalBlocked ?? 0,
        passRate: String(guardrailsData.passRate ?? 0),
        avgLatency: activeData.length
          ? Math.round(activeData.reduce((s, r) => s + (r.avgLatency ?? 0), 0) / activeData.length)
          : 0,
        count: activeData.length,
      };
    }
    return computeMetricsFromRows(activeData);
  }, [guardrailsData, activeData]);
  const chartData = guardrailsData?.chart;
  const sorted = useMemo(() => {
    return [...activeData].sort((a, b) => {
      const mult = sortDir === "desc" ? -1 : 1;
      const aVal = a[sortBy] ?? 0;
      const bVal = b[sortBy] ?? 0;
      return (Number(aVal) - Number(bVal)) * mult;
    });
  }, [activeData, sortBy, sortDir]);
  const isLoading = guardrailsLoading;
  const error = guardrailsError;

  const columns: ColumnDef<PerformanceRow>[] = [
    {
      header: "Guardrail",
      accessorKey: "name",
      enableSorting: false,
      cell: ({ row }) => (
        <button
          type="button"
          className="text-sm font-medium text-foreground hover:text-indigo-600 text-left"
          onClick={() => onSelectGuardrail(row.original.id)}
        >
          {row.original.name}
        </button>
      ),
    },
    {
      header: "Provider",
      accessorKey: "provider",
      enableSorting: false,
      cell: ({ row }) => (
        <span
          className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded border ${
            providerColors[row.original.provider] ?? providerColors.Custom
          }`}
        >
          {row.original.provider}
        </span>
      ),
    },
    {
      header: ({ column }) => <DataTableSortHeader column={column} title="Requests" />,
      accessorKey: "requestsEvaluated",
      meta: { numeric: true },
      sortDescFirst: false,
      cell: ({ row }) => row.original.requestsEvaluated.toLocaleString(),
    },
    {
      header: ({ column }) => <DataTableSortHeader column={column} title="Fail Rate" />,
      accessorKey: "failRate",
      meta: { numeric: true },
      sortDescFirst: false,
      cell: ({ row }) => (
        <span
          className={
            row.original.failRate > 15
              ? "text-destructive"
              : row.original.failRate > 5
                ? "text-warning"
                : "text-success"
          }
        >
          {row.original.failRate}%
          {row.original.trend === "up" && <span className="ml-1 text-xs text-destructive">↑</span>}
          {row.original.trend === "down" && <span className="ml-1 text-xs text-success">↓</span>}
        </span>
      ),
    },
    {
      header: ({ column }) => <DataTableSortHeader column={column} title="Avg. latency added" />,
      accessorKey: "avgLatency",
      meta: { numeric: true },
      sortDescFirst: false,
      cell: ({ row }) => (
        <span
          className={
            row.original.avgLatency == null
              ? "text-muted-foreground"
              : row.original.avgLatency > 150
                ? "text-destructive"
                : row.original.avgLatency > 50
                  ? "text-warning"
                  : "text-success"
          }
        >
          {row.original.avgLatency != null ? `${row.original.avgLatency}ms` : "—"}
        </span>
      ),
    },
    {
      header: "Status",
      accessorKey: "status",
      enableSorting: false,
      cell: ({ row }) => (
        <span className="inline-flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full ${
              row.original.status === "healthy"
                ? "bg-success"
                : row.original.status === "warning"
                  ? "bg-warning"
                  : "bg-destructive"
            }`}
          />
          <span className="text-xs text-muted-foreground capitalize">{row.original.status}</span>
        </span>
      ),
    },
  ];

  const sortableKeys: SortKey[] = ["failRate", "requestsEvaluated", "avgLatency"];
  const sorting = useMemo<SortingState>(() => [{ id: sortBy, desc: sortDir === "desc" }], [sortBy, sortDir]);
  const handleSortingChange: OnChangeFn<SortingState> = (updater) => {
    const nextSorting = typeof updater === "function" ? updater(sorting) : updater;
    const primarySort = nextSorting[0];
    if (primarySort && sortableKeys.includes(primarySort.id as SortKey)) {
      setSortBy(primarySort.id as SortKey);
      setSortDir(primarySort.desc ? "desc" : "asc");
    }
  };

  return (
    <div>
      <PageHeader
        icon={<HeartPulse />}
        title="Guardrails Monitor"
        subtitle="Monitor guardrail performance across all requests"
        utilities={
          <>
            {dateRangeControl}
            <Button variant="outline" title="Coming soon">
              <Download className="size-4" />
              Export Data
            </Button>
          </>
        }
      />

      <div className="mt-6 mb-6 grid grid-cols-[repeat(auto-fit,minmax(7rem,1fr))] gap-4">
        <MetricCard label="Total Evaluations" value={metrics.totalRequests.toLocaleString()} />
        <MetricCard
          label="Blocked Requests"
          value={metrics.totalBlocked.toLocaleString()}
          valueColor="text-destructive"
          icon={<TriangleAlert className="size-4 text-destructive" />}
        />
        <MetricCard
          label="Pass Rate"
          value={`${metrics.passRate}%`}
          valueColor="text-success"
          icon={<TrendingUp className="size-4 text-success" />}
        />
        <MetricCard
          label="Avg. latency added"
          value={`${metrics.avgLatency}ms`}
          valueColor={
            metrics.avgLatency > 150 ? "text-destructive" : metrics.avgLatency > 50 ? "text-warning" : "text-success"
          }
        />
        <MetricCard label="Active Guardrails" value={metrics.count} />
      </div>

      <div className="mb-6">
        <ScoreChart data={chartData} />
      </div>

      <div>
        {(isLoading || error) && (
          <div className="mb-2 flex items-center gap-2">
            {isLoading && (
              <span role="status" aria-busy="true" aria-label="Loading" className="inline-flex">
                <UiLoadingSpinner className="size-4 text-primary" />
              </span>
            )}
            {error && <span className="text-sm text-destructive">Failed to load data. Try again.</span>}
          </div>
        )}
        <DataTable
          columns={columns}
          data={sorted}
          getRowId={(row) => row.id}
          isLoading={isLoading}
          noDataMessage="No data for this period"
          onRowClick={(row) => onSelectGuardrail(row.id)}
          rowClassName={() => "cursor-pointer"}
          sortingMode="server"
          sorting={sorting}
          onSortingChange={handleSortingChange}
          enableSortingRemoval={false}
          size="compact"
          toolbar={() => (
            <div className="flex items-start justify-between gap-4">
              <div>
                <h5 className="mb-0 text-base font-semibold text-foreground">Guardrail Performance</h5>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Click a guardrail to view details, logs, and configuration
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setEvaluationModalOpen(true)}
                  title="Evaluation settings"
                >
                  <Settings className="size-4" />
                </Button>
              </div>
            </div>
          )}
        />
      </div>

      <EvaluationSettingsModal
        open={evaluationModalOpen}
        onClose={() => setEvaluationModalOpen(false)}
        accessToken={accessToken}
      />
    </div>
  );
}
