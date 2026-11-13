"use client";

import { useId, useMemo, useState, type ReactNode } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import type { DailyData } from "@/components/UsagePage/types";
import { StackedUsageChart } from "@/components/shared/charts";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatCompact, formatCompactUsd, modelColorMap } from "./overviewData";
import { ChartSkeleton, Panel, PANEL_INSET_X, Segmented } from "./Primitives";
import ModelContributionTable, { formatExactUsd } from "./ModelContributionTable";
import {
  bucketTooltipLabel,
  coveringRange,
  defaultGranularity,
  formatDayRange,
  usageTrend,
  type DayRange,
  type TrendGranularity,
  type TrendMetric,
} from "./usageTrendData";

type TrendView = "paired" | "requests";

const VIEW_OPTIONS = [
  { value: "paired", labelKey: "usage.trend.viewOptions.paired" },
  { value: "requests", labelKey: "usage.trend.viewOptions.requests" },
] as const satisfies readonly { value: TrendView; labelKey: ParseKeys }[];
const GRANULARITY_OPTIONS = [
  { value: "day", labelKey: "usage.breakdownChart.granularityOptions.day" },
  { value: "week", labelKey: "usage.breakdownChart.granularityOptions.week" },
  { value: "month", labelKey: "usage.trend.granularityMonth" },
] as const satisfies readonly { value: TrendGranularity; labelKey: ParseKeys }[];
const CHART_TITLE: Record<TrendMetric, ParseKeys> = {
  tokens: "usage.trend.tokensTitle",
  spend: "usage.trend.spendTitle",
  requests: "usage.trend.requestsTitle",
};

const MAX_LABELLED_BARS = 16;
const Y_AXIS_WIDTH = 64;

const exactFormat: Record<TrendMetric, (value: number) => string> = {
  spend: formatExactUsd,
  tokens: (value) => value.toLocaleString("en-US"),
  requests: (value) => value.toLocaleString("en-US"),
};
const compactFormat: Record<TrendMetric, (value: number) => string> = {
  spend: formatCompactUsd,
  tokens: formatCompact,
  requests: formatCompact,
};

function Legend({ labels, colors }: { labels: readonly string[]; colors: readonly string[] }) {
  const { t } = useTranslation();
  return (
    <ul aria-label={t("usage.trend.legend")} className="flex flex-wrap gap-x-4 gap-y-1.5">
      {labels.map((label, index) => (
        <li key={`${index}:${label}`} className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
          <span
            aria-hidden="true"
            className="size-2.5 shrink-0 rounded-[2px]"
            style={{ backgroundColor: colors[index] }}
          />
          <Tooltip>
            <TooltipTrigger className="max-w-48 truncate rounded-sm text-left outline-none focus-visible:ring-2 focus-visible:ring-ring/50">
              {label}
            </TooltipTrigger>
            <TooltipContent>{label}</TooltipContent>
          </Tooltip>
        </li>
      ))}
    </ul>
  );
}

function Notice({ children }: { children: ReactNode }) {
  return <p className="text-xs leading-5 text-muted-foreground">{children}</p>;
}

interface UsageTrendPanelProps {
  results: readonly DailyData[];
  range: DayRange | null;
  loading: boolean;
  failed: boolean;
}

/** Tokens over spend per bucket, stacked by one shared set of leading models, then the exact table behind both. */
export default function UsageTrendPanel({ results, range, loading, failed }: UsageTrendPanelProps) {
  const { t } = useTranslation();
  const syncId = useId();
  const [view, setView] = useState<TrendView>("paired");
  const [chosenGranularity, setChosenGranularity] = useState<TrendGranularity | null>(null);
  const covered = useMemo(() => (range ? coveringRange(range, results) : null), [range, results]);
  const granularity = chosenGranularity ?? (covered ? defaultGranularity(covered) : "day");
  const colorByModel = useMemo(() => modelColorMap(results), [results]);
  const trend = useMemo(
    () => (range ? usageTrend({ results, range, granularity, colorByModel }) : null),
    [results, range, granularity, colorByModel],
  );
  const bucketByKey = useMemo(() => new Map((trend?.buckets ?? []).map((bucket) => [bucket.key, bucket])), [trend]);
  const metrics: readonly TrendMetric[] = view === "paired" ? ["tokens", "spend"] : ["requests"];
  const partialBuckets = granularity === "day" ? [] : (trend?.buckets ?? []).filter((bucket) => bucket.partial);
  const showTotals = (trend?.buckets.length ?? 0) <= MAX_LABELLED_BARS;
  const legendSeries = trend?.series.tokens;
  const empty = !loading && !failed && results.length === 0;

  const body = () => {
    if (failed) return <Notice>{t("usage.trend.unavailable")}</Notice>;
    if (loading || trend === null) {
      return (
        <div role="status" aria-label={t("usage.trend.loading")} className="grid gap-4">
          <ChartSkeleton className="h-60" />
          {view === "paired" && <ChartSkeleton className="h-60" />}
        </div>
      );
    }
    if (empty) return <p className="py-10 text-center text-sm text-muted-foreground">{t("usage.trend.empty")}</p>;
    return (
      <div className="grid gap-4">
        {legendSeries && <Legend labels={legendSeries.labels} colors={legendSeries.colors} />}
        {metrics.map((metric) => {
          const series = trend.series[metric];
          const totals = trend.totals[metric];
          return (
            <figure key={metric} className="grid gap-1">
              <figcaption className="text-xs font-medium text-muted-foreground">{t(CHART_TITLE[metric])}</figcaption>
              <StackedUsageChart
                data={series.data}
                series={series.keys}
                labels={series.labels}
                colors={series.colors}
                xKey="date"
                xLabel={(key) => series.data.find((row) => row.date === key)?.label ?? key}
                tooltipLabel={(key) => {
                  const bucket = bucketByKey.get(key);
                  return bucket ? bucketTooltipLabel(bucket, granularity) : key;
                }}
                format={compactFormat[metric]}
                tooltipFormat={exactFormat[metric]}
                totalFor={(key) => totals.get(key)}
                showTotals={showTotals}
                syncId={syncId}
                yAxisWidth={Y_AXIS_WIDTH}
                ariaLabel={t("usage.trend.chartLabel", { metric: t(CHART_TITLE[metric]) })}
                className="h-60"
              />
            </figure>
          );
        })}
        {partialBuckets.length > 0 && (
          <Notice>
            {t("usage.trend.partialNote", {
              periods: partialBuckets.map((bucket) => formatDayRange(bucket)).join(", "),
            })}
          </Notice>
        )}
        {(trend.unattributed.spend !== 0 || trend.unattributed.tokens !== 0) && (
          <Notice>
            {t("usage.trend.unattributedNote", {
              spend: formatExactUsd(trend.unattributed.spend),
              tokens: trend.unattributed.tokens.toLocaleString("en-US"),
            })}
          </Notice>
        )}
      </div>
    );
  };

  return (
    <Panel
      testId="usage-trend"
      title={t("usage.trend.title")}
      subtitle={t("usage.trend.subtitle", { range: range ? formatDayRange(range) : "" })}
      action={
        <>
          <Segmented
            label={t("usage.trend.viewLabel")}
            value={view}
            options={VIEW_OPTIONS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }))}
            onChange={setView}
          />
          <Segmented
            label={t("usage.breakdownChart.bucketSize")}
            value={granularity}
            options={GRANULARITY_OPTIONS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }))}
            onChange={setChosenGranularity}
          />
        </>
      }
      bodyClassName="px-0 pt-4 pb-0"
    >
      <div className={PANEL_INSET_X}>{body()}</div>
      {!failed && !loading && !empty && trend !== null && (
        <section aria-label={t("usage.trend.table.title")} className={`mt-4 border-t pt-3 pb-2 ${PANEL_INSET_X}`}>
          <h4 className="text-sm leading-5 font-medium text-foreground">{t("usage.trend.table.title")}</h4>
          <ModelContributionTable rows={trend.rows} total={trend.total} />
        </section>
      )}
    </Panel>
  );
}
