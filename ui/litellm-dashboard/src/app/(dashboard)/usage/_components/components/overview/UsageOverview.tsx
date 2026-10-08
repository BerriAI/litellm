"use client";

/**
 * Usage overview: a stat strip, the Model Leaderboard's stacked chart and
 * tooltip applied to the usage aggregate, then a ranked share list in
 * matching colors.
 */

import React, { type ReactNode, useMemo, useState } from "react";
import type { DailyData } from "@/components/UsagePage/types";
import { KeyRound } from "lucide-react";
import { StackedUsageChart, type StackedUsageScale } from "@/components/shared/charts";
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
  loading,
  requestCountsPending,
  topKeys,
  gatewayByEndpoint,
  topAgents,
  providerBreakdown,
  budget,
}: UsageOverviewProps) {
  const [state, setState] = useState<BreakdownState>({ metric: "spend", dimension: "model_groups" });
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
              <span>
                <span className={totals.failed > 0 ? "text-destructive tabular-nums" : "tabular-nums"}>
                  {totals.failed.toLocaleString()}
                </span>{" "}
                failed
              </span>
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
