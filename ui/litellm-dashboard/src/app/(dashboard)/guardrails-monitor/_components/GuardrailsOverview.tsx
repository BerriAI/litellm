import type { ColumnDef, OnChangeFn, SortingState } from "@tanstack/react-table";
import { CircleDollarSign, Download, HeartPulse, Settings, TrendingUp, TriangleAlert } from "lucide-react";
import React, { useMemo, useState } from "react";
import { DataTable, DataTableSortHeader } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells/money_cell";
import { CellTooltip } from "@/components/shared/table_cells/cell_tooltip";
import {
  type GuardrailUsageOverviewRow,
  useGuardrailsUsageOverview,
} from "@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage";
import { CalcPopover, MathTable } from "@/components/GuardrailsMonitor/CalcPopover";
import { UnpricedNote } from "@/components/GuardrailsMonitor/UnpricedNote";
import {
  counterLabel,
  formatCost,
  totalUnits,
  unpricedSummary,
  type UsageUnits,
} from "@/components/GuardrailsMonitor/usageUnits";
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

type SortKey = "failRate" | "requestsEvaluated" | "avgLatency" | "cost";

const providerColors: Record<string, string> = {
  Bedrock: "bg-warning/15 text-warning border-warning/20",
  "Google Cloud": "bg-info/15 text-info border-info/20",
  LiteLLM:
    "bg-indigo-100 text-indigo-700 border-indigo-200 dark:bg-indigo-950 dark:text-indigo-300 dark:border-indigo-800",
  Custom: "bg-muted text-muted-foreground border-border",
};

const EMPTY_METRICS = {
  totalRequests: 0,
  totalBlocked: 0,
  passRate: "0",
  avgLatency: 0,
  count: 0,
  totalCost: null as number | null,
  untracked: {} as UsageUnits,
};

function UsageUnitsCell({ units }: { units: GuardrailUsageOverviewRow["usageUnits"] }) {
  const counters = Object.entries(units);
  if (counters.length === 0) return <span className="text-muted-foreground">—</span>;
  return (
    <CellTooltip
      content={
        <ul className="space-y-0.5">
          {counters.map(([counter, n]) => (
            <li key={counter}>
              {counterLabel(counter)}: {n.toLocaleString()}
            </li>
          ))}
        </ul>
      }
      trigger={<span className="tabular-nums">{totalUnits(units).toLocaleString()}</span>}
    />
  );
}

function TotalCostMath({
  rows,
  total,
  untracked,
}: {
  rows: GuardrailUsageOverviewRow[];
  total: number | null;
  untracked: UsageUnits;
}) {
  return (
    <CalcPopover title="How this cost is calculated" formula="guardrail + guardrail + … = guardrail cost">
      <MathTable
        rows={rows
          .filter((row) => row.cost != null)
          .map((row) => ({ label: row.name, parts: [formatCost(row.cost)], note: null }))}
        total={formatCost(total)}
      />
      <p className="text-xs text-muted-foreground">
        {`Each guardrail's cost is its units per counter × that counter's per-unit price from the cost map. Open a guardrail for its per-counter math.`}
      </p>
      <UnpricedNote unpriced={untracked} />
    </CalcPopover>
  );
}

function CostCell({ row }: { row: GuardrailUsageOverviewRow }) {
  const unpriced = unpricedSummary(row.untrackedUsageUnits);
  return (
    <span className="inline-flex w-full items-center justify-end gap-1">
      {unpriced && (
        <CellTooltip
          content={`${unpriced}: these units have no known price and are left out of the cost`}
          trigger={<TriangleAlert aria-label={unpriced} className="size-3.5 shrink-0 text-warning" />}
        />
      )}
      <MoneyCell value={row.cost} emptyText="—" showZero />
    </span>
  );
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
  } = useGuardrailsUsageOverview({ accessToken, startDate, endDate });

  const activeData: GuardrailUsageOverviewRow[] = useMemo(() => guardrailsData?.rows ?? [], [guardrailsData]);
  const metrics = useMemo(() => {
    if (!guardrailsData) return EMPTY_METRICS;
    return {
      totalRequests: guardrailsData.totalRequests,
      totalBlocked: guardrailsData.totalBlocked,
      passRate: String(guardrailsData.passRate),
      avgLatency: activeData.length
        ? Math.round(activeData.reduce((s, r) => s + (r.avgLatency ?? 0), 0) / activeData.length)
        : 0,
      count: activeData.length,
      totalCost: guardrailsData.totalCost,
      untracked: guardrailsData.totalUntrackedUsageUnits,
    };
  }, [guardrailsData, activeData]);
  const chartData = guardrailsData?.chart;
  const sorted = useMemo(() => {
    const mult = sortDir === "desc" ? -1 : 1;
    return [...activeData].sort((a, b) => {
      const aVal = a[sortBy];
      const bVal = b[sortBy];
      if (aVal == null || bVal == null) return Number(aVal == null) - Number(bVal == null);
      return (aVal - bVal) * mult;
    });
  }, [activeData, sortBy, sortDir]);
  const isLoading = guardrailsLoading;
  const error = guardrailsError;

  const columns: ColumnDef<GuardrailUsageOverviewRow>[] = [
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
      header: "Usage Units",
      accessorKey: "usageUnits",
      enableSorting: false,
      meta: { numeric: true },
      cell: ({ row }) => <UsageUnitsCell units={row.original.usageUnits} />,
    },
    {
      header: ({ column }) => <DataTableSortHeader column={column} title="Cost" />,
      accessorKey: "cost",
      meta: { numeric: true },
      sortDescFirst: false,
      cell: ({ row }) => <CostCell row={row.original} />,
    },
  ];

  const sortableKeys: SortKey[] = ["failRate", "requestsEvaluated", "avgLatency", "cost"];
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
        <MetricCard
          label="Guardrail Cost"
          value={formatCost(metrics.totalCost)}
          valueColor={metrics.totalCost != null ? "text-foreground" : "text-muted-foreground"}
          icon={<CircleDollarSign className="size-4" />}
          subtitle={unpricedSummary(metrics.untracked) ?? undefined}
          hint={<TotalCostMath rows={activeData} total={metrics.totalCost} untracked={metrics.untracked} />}
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
