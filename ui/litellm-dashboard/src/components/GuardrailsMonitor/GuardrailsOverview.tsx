import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Download,
  Settings,
  Shield,
  TrendingUp,
} from "lucide-react";
import React, { useMemo, useState } from "react";
import { getGuardrailsUsageOverview } from "@/components/networking";
import { type PerformanceRow } from "./mockData";
import { EvaluationSettingsModal } from "./EvaluationSettingsModal";
import { MetricCard } from "./MetricCard";
import { ScoreChart } from "./ScoreChart";

interface GuardrailsOverviewProps {
  accessToken?: string | null;
  startDate: string;
  endDate: string;
  onSelectGuardrail: (id: string) => void;
}

type SortKey =
  | "failRate"
  | "requestsEvaluated"
  | "avgLatency"
  | "falsePositiveRate"
  | "falseNegativeRate";

const providerColors: Record<string, string> = {
  Bedrock:
    "bg-orange-100 text-orange-700 border-orange-200 dark:bg-orange-950/30 dark:text-orange-300 dark:border-orange-800",
  "Google Cloud":
    "bg-sky-100 text-sky-700 border-sky-200 dark:bg-sky-950/30 dark:text-sky-300 dark:border-sky-800",
  LiteLLM:
    "bg-indigo-100 text-indigo-700 border-indigo-200 dark:bg-indigo-950/30 dark:text-indigo-300 dark:border-indigo-800",
  Custom: "bg-muted text-muted-foreground border-border",
};

function computeMetricsFromRows(data: PerformanceRow[]) {
  const totalRequests = data.reduce((sum, r) => sum + r.requestsEvaluated, 0);
  const totalBlocked = data.reduce(
    (sum, r) => sum + Math.round((r.requestsEvaluated * r.failRate) / 100),
    0,
  );
  const passRate =
    totalRequests > 0
      ? ((1 - totalBlocked / totalRequests) * 100).toFixed(1)
      : "0";
  const withLat = data.filter((r) => r.avgLatency != null);
  const avgLatency =
    withLat.length > 0
      ? Math.round(
          withLat.reduce((sum, r) => sum + (r.avgLatency ?? 0), 0) /
            withLat.length,
        )
      : 0;
  return {
    totalRequests,
    totalBlocked,
    passRate,
    avgLatency,
    count: data.length,
  };
}

export function GuardrailsOverview({
  accessToken = null,
  startDate,
  endDate,
  onSelectGuardrail,
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
          ? Math.round(
              activeData.reduce((s, r) => s + (r.avgLatency ?? 0), 0) /
                activeData.length,
            )
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

  const toggleSort = (key: SortKey) => {
    if (sortBy === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortBy(key);
      setSortDir("desc");
    }
  };

  return (
    <div>
      <div className="flex items-start justify-between mb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Shield className="h-5 w-5 text-indigo-500" />
            <h1 className="text-xl font-semibold text-foreground">
              Guardrails Monitor
            </h1>
          </div>
          <p className="text-sm text-muted-foreground">
            Monitor guardrail performance across all requests
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" title="Coming soon">
            <Download className="h-4 w-4" />
            Export Data
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
        <MetricCard
          label="Total Evaluations"
          value={metrics.totalRequests.toLocaleString()}
        />
        <MetricCard
          label="Blocked Requests"
          value={metrics.totalBlocked.toLocaleString()}
          valueColor="text-red-600 dark:text-red-400"
          icon={<AlertTriangle className="h-4 w-4 text-red-400" />}
        />
        <MetricCard
          label="Pass Rate"
          value={`${metrics.passRate}%`}
          valueColor="text-emerald-600 dark:text-emerald-400"
          icon={<TrendingUp className="h-4 w-4 text-emerald-400" />}
        />
        <MetricCard
          label="Avg. latency added"
          value={`${metrics.avgLatency}ms`}
          valueColor={
            metrics.avgLatency > 150
              ? "text-red-600 dark:text-red-400"
              : metrics.avgLatency > 50
                ? "text-amber-600 dark:text-amber-400"
                : "text-emerald-600 dark:text-emerald-400"
          }
        />
        <MetricCard label="Active Guardrails" value={metrics.count} />
      </div>

      <div className="mb-6">
        <ScoreChart data={chartData} />
      </div>

      <Card className="border border-border rounded-lg bg-background p-0">
        {(isLoading || error) && (
          <div className="px-6 py-4 border-b border-border flex items-center gap-2">
            {isLoading && <Skeleton className="h-4 w-32" />}
            {error && (
              <span className="text-sm text-destructive">
                Failed to load data. Try again.
              </span>
            )}
          </div>
        )}
        <div className="px-6 py-4 border-b border-border flex items-start justify-between gap-4">
          <div>
            <h5 className="text-base font-semibold text-foreground !mb-0">
              Guardrail Performance
            </h5>
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
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Guardrail</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead
                className="text-right cursor-pointer"
                onClick={() => toggleSort("requestsEvaluated")}
              >
                Requests{" "}
                {sortBy === "requestsEvaluated"
                  ? sortDir === "desc"
                    ? "↓"
                    : "↑"
                  : ""}
              </TableHead>
              <TableHead
                className="text-right cursor-pointer"
                onClick={() => toggleSort("failRate")}
              >
                Fail Rate{" "}
                {sortBy === "failRate" ? (sortDir === "desc" ? "↓" : "↑") : ""}
              </TableHead>
              <TableHead
                className="text-right cursor-pointer"
                onClick={() => toggleSort("avgLatency")}
              >
                Avg. latency added{" "}
                {sortBy === "avgLatency"
                  ? sortDir === "desc"
                    ? "↓"
                    : "↑"
                  : ""}
              </TableHead>
              <TableHead className="text-center">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.length === 0 && !isLoading ? (
              <TableRow>
                <TableCell
                  colSpan={6}
                  className="text-center text-muted-foreground py-8"
                >
                  No data for this period
                </TableCell>
              </TableRow>
            ) : (
              sorted.map((row) => (
                <TableRow
                  key={row.id}
                  className="cursor-pointer"
                  onClick={() => onSelectGuardrail(row.id)}
                >
                  <TableCell>
                    <button
                      type="button"
                      className="text-sm font-medium text-foreground hover:text-primary text-left"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectGuardrail(row.id);
                      }}
                    >
                      {row.name}
                    </button>
                  </TableCell>
                  <TableCell>
                    <span
                      className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded border ${
                        providerColors[row.provider] ?? providerColors.Custom
                      }`}
                    >
                      {row.provider}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    {row.requestsEvaluated.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right">
                    <span
                      className={
                        row.failRate > 15
                          ? "text-red-600 dark:text-red-400"
                          : row.failRate > 5
                            ? "text-amber-600 dark:text-amber-400"
                            : "text-emerald-600 dark:text-emerald-400"
                      }
                    >
                      {row.failRate}%
                      {row.trend === "up" && (
                        <span className="ml-1 text-xs text-red-400">↑</span>
                      )}
                      {row.trend === "down" && (
                        <span className="ml-1 text-xs text-emerald-400">↓</span>
                      )}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span
                      className={
                        row.avgLatency == null
                          ? "text-muted-foreground"
                          : row.avgLatency > 150
                            ? "text-red-600 dark:text-red-400"
                            : row.avgLatency > 50
                              ? "text-amber-600 dark:text-amber-400"
                              : "text-emerald-600 dark:text-emerald-400"
                      }
                    >
                      {row.avgLatency != null ? `${row.avgLatency}ms` : "—"}
                    </span>
                  </TableCell>
                  <TableCell className="text-center">
                    <span className="inline-flex items-center gap-1.5">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          row.status === "healthy"
                            ? "bg-emerald-500"
                            : row.status === "warning"
                              ? "bg-amber-500"
                              : "bg-red-500"
                        }`}
                      />
                      <span className="text-xs text-muted-foreground capitalize">
                        {row.status}
                      </span>
                    </span>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>

      <EvaluationSettingsModal
        open={evaluationModalOpen}
        onClose={() => setEvaluationModalOpen(false)}
        accessToken={accessToken}
      />
    </div>
  );
}
