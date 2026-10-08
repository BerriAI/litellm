"use client";

/**
 * Usage overview: a stat strip, the Model Leaderboard's stacked chart and
 * tooltip applied to the usage aggregate, then a ranked share list in
 * matching colors.
 */

import React, { type ReactNode, useMemo, useState } from "react";
import type { DailyData } from "@/components/UsagePage/types";
import { ChevronDown, ChevronRight, Info, KeyRound } from "lucide-react";
import { StackedUsageChart, type StackedUsageScale } from "@/components/shared/charts";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { FailedRequestBreakdown } from "../gatewayActivity";
import {
  bucketSeries,
  bucketTotals,
  labelForDate,
  dailyTotals,
  formatCompact,
  formatLatency,
  formatMetricValue,
  formatUsd,
  type Granularity,
  type OverviewTotals,
  type UsageMetric,
} from "./overviewData";
import { BreakdownControls, Leaderboard, useBreakdown, type BreakdownState } from "./BreakdownChart";
import { cn } from "@/lib/cva.config";
import { ChartSkeleton, Panel, PANEL_INSET_X, Segmented, Sparkline, Stat } from "./Primitives";

interface UsageOverviewProps {
  results: readonly DailyData[];
  totals: OverviewTotals;
  failureBreakdown: FailedRequestBreakdown | null;
  loading: boolean;
  requestCountsPending: boolean;
  budget: number | null;
  topKeys: ReactNode;
  gatewayByEndpoint: ReactNode;
  topAgents: ReactNode;
  providerBreakdown: ReactNode;
}

const METRIC_NOUN: Record<UsageMetric, string> = { spend: "spend", tokens: "tokens", requests: "requests" };
const BRAND = "#2b3fd6";
const GRANULARITY_OPTIONS = [
  { value: "day", label: "Daily" },
  { value: "week", label: "Weekly" },
] as const satisfies readonly { value: Granularity; label: string }[];
const SCALE_OPTIONS = [
  { value: "linear", label: "Linear" },
  { value: "log", label: "Log" },
] as const satisfies readonly { value: StackedUsageScale; label: string }[];

export default function UsageOverview({
  results,
  totals,
  failureBreakdown,
  loading,
  requestCountsPending,
  topKeys,
  gatewayByEndpoint,
  topAgents,
  providerBreakdown,
  budget,
}: UsageOverviewProps) {
  const [state, setState] = useState<BreakdownState>({ metric: "spend", dimension: "model_groups" });
  const [showFailedBreakdown, setShowFailedBreakdown] = useState(false);
  const [granularity, setGranularity] = useState<Granularity>("day");
  const [scale, setScale] = useState<StackedUsageScale>("linear");
  const { series: dailySeries, ranking } = useBreakdown(results, state, 8);
  const series = useMemo(() => bucketSeries(dailySeries, granularity), [dailySeries, granularity]);
  const totalsByBucket = useMemo(() => bucketTotals(series), [series]);
  const daily = useMemo(() => dailyTotals(results), [results]);
  const format = (value: number) => formatMetricValue(value, state.metric);

  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-1 overflow-hidden rounded-xl border bg-card sm:grid-cols-2 lg:grid-cols-4 lg:divide-x">
        <StatCell trend={<Sparkline data={daily} dataKey="spend" color={BRAND} className="h-14" />}>
          <Stat
            label="Spend"
            value={formatUsd(totals.spend)}
            pending={loading}
            hint={budget ? `of ${formatUsd(budget)} budget` : "No budget set"}
          />
        </StatCell>
        <StatCell trend={<Sparkline data={daily} dataKey="requests" color={BRAND} className="h-14" />}>
          <Stat label="Total Requests" value={totals.requests.toLocaleString()} pending={requestCountsPending} />
          {!requestCountsPending && (
            <div className="mt-0.5 flex gap-3 text-xs leading-4 text-muted-foreground">
              <span>
                <span className="text-success tabular-nums">{totals.successful.toLocaleString()}</span> ok
              </span>
              {failureBreakdown ? (
                <button
                  type="button"
                  aria-label={`Failed Requests: ${totals.failed.toLocaleString()}`}
                  aria-expanded={showFailedBreakdown}
                  aria-controls="gateway-failure-status-breakdown"
                  className="inline-flex items-center gap-1 rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  onClick={() => setShowFailedBreakdown(!showFailedBreakdown)}
                >
                  <span className={totals.failed > 0 ? "text-destructive tabular-nums" : "tabular-nums"}>
                    {totals.failed.toLocaleString()}
                  </span>{" "}
                  failed
                  {showFailedBreakdown ? (
                    <ChevronDown className="size-3 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="size-3 text-muted-foreground" />
                  )}
                </button>
              ) : (
                <span>
                  <span className={totals.failed > 0 ? "text-destructive tabular-nums" : "tabular-nums"}>
                    {totals.failed.toLocaleString()}
                  </span>{" "}
                  failed
                </span>
              )}
            </div>
          )}
          {failureBreakdown && (
            <div className="mt-2 space-y-1 text-sm text-muted-foreground">
              <p>Client errors (4xx): {failureBreakdown.clientErrors.toLocaleString()}</p>
              <p>Server errors (5xx): {failureBreakdown.serverErrors.toLocaleString()}</p>
            </div>
          )}
        </StatCell>
        <StatCell trend={<Sparkline data={daily} dataKey="tokens" color={BRAND} className="h-14" />}>
          <Stat
            label="Total Tokens"
            value={loading ? null : formatCompact(totals.tokens)}
            exact={`${totals.tokens.toLocaleString()} tokens`}
            pending={loading}
            hint={loading ? undefined : `${formatCompact(totals.cacheReadTokens)} cache read`}
          />
        </StatCell>
        <StatCell>
          <div className="grid gap-3">
            <Stat label="Avg cost / request" value={formatUsd(totals.avgCostPerRequest, 4)} pending={loading} />
            <Stat
              label="Avg latency"
              value={totals.avgLatencyMs === null ? "—" : formatLatency(totals.avgLatencyMs)}
              pending={loading}
              hint={totals.successRate === null ? undefined : `${totals.successRate.toFixed(1)}% success rate`}
            />
          </div>
        </StatCell>
      </div>

      {showFailedBreakdown && failureBreakdown && (
        <div id="gateway-failure-status-breakdown" className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {failureBreakdown.rows.map((row) => (
            <div key={row.status_code} className="rounded-xl border bg-card p-3">
              <h3 className="text-sm font-medium text-foreground">
                {row.status_code} {row.label}
              </h3>
              <p className="mt-2 text-2xl font-bold text-destructive">{row.failed_requests.toLocaleString()}</p>
            </div>
          ))}
          {failureBreakdown.notRecorded > 0 && (
            <div className="rounded-xl border bg-card p-3">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-medium text-foreground">Not recorded</h3>
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <button
                        type="button"
                        aria-label="Not recorded failure details"
                        className="inline-flex items-center rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      >
                        <Info aria-hidden="true" className="size-4 text-muted-foreground hover:text-foreground" />
                      </button>
                    }
                  />
                  <TooltipContent>
                    Failed before per-status tracking was available, or counted by an older proxy version
                  </TooltipContent>
                </Tooltip>
              </div>
              <p className="mt-2 text-2xl font-bold text-destructive">
                {failureBreakdown.notRecorded.toLocaleString()}
              </p>
            </div>
          )}
        </div>
      )}

      <Panel
        title="Top models"
        subtitle={`${granularity === "day" ? "Daily" : "Weekly"} ${METRIC_NOUN[state.metric]}, top 8 stacked`}
        action={
          <>
            <BreakdownControls state={state} onChange={setState} showDimension={false} />
            <Segmented
              label="Bucket size"
              value={granularity}
              options={GRANULARITY_OPTIONS}
              onChange={setGranularity}
            />
            <Segmented label="Scale" value={scale} options={SCALE_OPTIONS} onChange={setScale} />
          </>
        }
        bodyClassName="px-0 pt-4 pb-0"
      >
        <div className="px-2 pb-2">
          {loading ? (
            <ChartSkeleton className="mx-3 h-[380px] w-auto" />
          ) : (
            <StackedUsageChart
              data={series.data}
              series={series.keys}
              labels={series.labels}
              colors={series.colors}
              xKey="date"
              xLabel={(date) => labelForDate(series, date)}
              scale={scale}
              format={format}
              totalFor={(date) => totalsByBucket.get(date)}
            />
          )}
        </div>
        <div className={cn("border-t py-2", PANEL_INSET_X)}>
          <Leaderboard
            ranking={ranking}
            series={dailySeries}
            metric={state.metric}
            dimension={state.dimension}
            columns={2}
            limit={10}
          />
        </div>
      </Panel>

      {topAgents}
      {providerBreakdown}

      <Panel icon={KeyRound} title="Top Virtual Keys">
        {topKeys}
      </Panel>
      {gatewayByEndpoint}
    </div>
  );
}

function StatCell({ children, trend }: { children: React.ReactNode; trend?: React.ReactNode }) {
  return (
    <div
      className={cn(
        "flex min-w-0 flex-col justify-between gap-3 border-b py-4 last:border-b-0 lg:border-b-0",
        PANEL_INSET_X,
      )}
    >
      <div className="min-w-0">{children}</div>
      {trend}
    </div>
  );
}
