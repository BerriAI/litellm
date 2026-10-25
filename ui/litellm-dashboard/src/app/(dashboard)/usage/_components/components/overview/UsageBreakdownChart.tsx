"use client";

import { useMemo, useState } from "react";
import type { DailyData } from "@/components/UsagePage/types";
import { StackedUsageChart, type StackedUsageScale } from "@/components/shared/charts";
import { cn } from "@/lib/cva.config";
import { useLanguage } from "@/contexts/LanguageContext";
import { BreakdownControls, Leaderboard, useBreakdown, type BreakdownState } from "./BreakdownChart";
import {
  bucketSeries,
  bucketTotals,
  formatMetricValue,
  labelForDate,
  modelColorMap,
  type BreakdownDimension,
  type Granularity,
  type UsageMetric,
} from "./overviewData";
import { ChartSkeleton, Panel, PANEL_INSET_X, Segmented } from "./Primitives";

const METRIC_NOUN: Record<UsageMetric, string> = { spend: "spend", tokens: "tokens", requests: "requests" };
const GRANULARITY_OPTIONS = [
  { value: "day", label: "Daily" },
  { value: "week", label: "Weekly" },
] as const satisfies readonly { value: Granularity; label: string }[];
const SCALE_OPTIONS = [
  { value: "linear", label: "Linear" },
  { value: "log", label: "Log" },
] as const satisfies readonly { value: StackedUsageScale; label: string }[];

interface UsageBreakdownChartProps {
  results: readonly DailyData[];
  loading: boolean;
  compact?: boolean;
  colorByModel?: ReadonlyMap<string, string>;
  dimension?: BreakdownDimension;
}

export default function UsageBreakdownChart({
  results,
  loading,
  compact = false,
  colorByModel,
  dimension = "model_groups",
}: UsageBreakdownChartProps) {
  const { t, language } = useLanguage();
  const [metric, setMetric] = useState<UsageMetric>("spend");
  const state: BreakdownState = { metric, dimension };
  const [granularity, setGranularity] = useState<Granularity>("day");
  const [scale, setScale] = useState<StackedUsageScale>("linear");
  const modelColors = useMemo(() => colorByModel ?? modelColorMap(results), [colorByModel, results]);
  const { series: dailySeries, ranking } = useBreakdown(results, state, {
    top: 8,
    colorByKey: modelColors,
    otherLabel: t("Other"),
  });
  const series = useMemo(() => bucketSeries(dailySeries, granularity), [dailySeries, granularity]);
  const totalsByBucket = useMemo(() => bucketTotals(series), [series]);
  const format = (value: number) => formatMetricValue(value, state.metric);
  const xLabel = (date: string) => {
    if (language !== "zh-CN") return labelForDate(series, date);
    return granularity === "week" ? t("Week of {date}", { date }) : date;
  };

  return (
    <Panel
      title={t("{period} usage", { period: t(granularity === "day" ? "Daily" : "Weekly") })}
      subtitle={t("{period} {metric} by model (top 8, rest grouped as Other)", {
        period: t(granularity === "day" ? "Daily" : "Weekly"),
        metric: t(METRIC_NOUN[state.metric]),
      })}
      action={
        <>
          <BreakdownControls state={state} onChange={(next) => setMetric(next.metric)} showDimension={false} />
          <Segmented
            label={t("Bucket size")}
            value={granularity}
            options={GRANULARITY_OPTIONS.map((option) => ({ ...option, label: t(option.label) }))}
            onChange={setGranularity}
          />
          <Segmented
            label={t("Scale")}
            value={scale}
            options={SCALE_OPTIONS.map((option) => ({ ...option, label: t(option.label) }))}
            onChange={setScale}
          />
        </>
      }
      bodyClassName="px-0 pt-4 pb-0"
    >
      <div className="px-2 pb-2">
        {loading ? (
          <ChartSkeleton className={cn("mx-3 w-auto", compact ? "h-56" : "h-[380px]")} />
        ) : (
          <StackedUsageChart
            data={series.data}
            series={series.keys}
            labels={series.labels}
            colors={series.colors}
            xKey="date"
            xLabel={xLabel}
            scale={scale}
            format={format}
            totalFor={(date) => totalsByBucket.get(date)}
            className={compact ? "h-56" : undefined}
          />
        )}
      </div>
      <div className={cn("border-t pt-3 pb-2", PANEL_INSET_X)}>
        <h4 className="text-sm leading-5 font-medium text-foreground">{t("Top models")}</h4>
        {loading ? (
          <div role="status" aria-label={t("Loading top models")}>
            <ChartSkeleton className="my-2 h-20" />
          </div>
        ) : (
          <Leaderboard
            ranking={ranking}
            series={dailySeries}
            metric={state.metric}
            dimension={state.dimension}
            colorByKey={modelColors}
            columns={compact ? 1 : 2}
            limit={10}
          />
        )}
      </div>
    </Panel>
  );
}
