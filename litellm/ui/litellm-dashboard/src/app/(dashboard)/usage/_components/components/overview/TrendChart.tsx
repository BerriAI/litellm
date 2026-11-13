"use client";

import type { ReactNode } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { CircleHelp } from "lucide-react";
import { StackedUsageChart } from "@/components/shared/charts";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { COMPACT_FORMAT, EXACT_FORMAT } from "./trendFormat";
import type { TrendSelection } from "./trendSelection";
import { bucketTooltipLabel, formatDayRange, type TrendGranularity, type usageTrend } from "./usageTrendData";

export type Trend = ReturnType<typeof usageTrend>;

const Y_AXIS_WIDTH = 64;

export const GRANULARITY_ADJECTIVE: Readonly<Record<TrendGranularity, ParseKeys>> = {
  day: "usage.breakdownChart.granularity.daily",
  week: "usage.breakdownChart.granularity.weekly",
  month: "usage.trend.granularityMonth",
};

export const METRIC_NOUN: Readonly<Record<TrendSelection["metric"], ParseKeys>> = {
  tokens: "usage.breakdownChart.metricNoun.tokens",
  spend: "usage.breakdownChart.metricNoun.spend",
  requests: "usage.breakdownChart.metricNoun.requests",
};

export function TrendNotice({ children }: { children: ReactNode }) {
  return <p className="text-xs leading-5 text-muted-foreground">{children}</p>;
}

export function TrendLegend({
  title,
  hint,
  labels,
  colors,
}: {
  title: string;
  hint: string;
  labels: readonly string[];
  colors: readonly string[];
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      <span className="flex items-center gap-1 text-xs font-medium text-foreground">
        {title}
        <Tooltip>
          <TooltipTrigger
            aria-label={hint}
            className="rounded-sm text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            <CircleHelp aria-hidden="true" className="size-3.5" />
          </TooltipTrigger>
          <TooltipContent>{hint}</TooltipContent>
        </Tooltip>
      </span>
      <ul aria-label={title} className="flex min-w-0 flex-wrap gap-x-4 gap-y-1.5">
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
    </div>
  );
}

export function PartialPeriodNote({ trend, granularity }: { trend: Trend; granularity: TrendGranularity }) {
  const { t } = useTranslation();
  const partial = granularity === "day" ? [] : trend.buckets.filter((bucket) => bucket.partial);
  if (partial.length === 0) return null;
  return (
    <TrendNotice>
      {t("usage.trend.partialNote", { periods: partial.map((bucket) => formatDayRange(bucket)).join(", ") })}
    </TrendNotice>
  );
}

export default function TrendChart({
  trend,
  selection,
  labels,
  ariaLabel,
  className,
}: {
  trend: Trend;
  selection: TrendSelection;
  labels?: readonly string[];
  ariaLabel: string;
  className: string;
}) {
  const { metric, granularity } = selection;
  const series = trend.series[metric];
  const totals = trend.totals[metric];
  const bucketByKey = new Map(trend.buckets.map((bucket) => [bucket.key, bucket]));
  const tickByKey = new Map(series.data.map((point) => [point.date, point.label]));
  const tooltipLabel = (key: string) => {
    const bucket = bucketByKey.get(key);
    return bucket ? bucketTooltipLabel(bucket, granularity) : key;
  };
  return (
    <StackedUsageChart
      data={series.data}
      series={series.keys}
      labels={labels ?? series.labels}
      colors={series.colors}
      xKey="date"
      xLabel={(key) => tickByKey.get(key) ?? key}
      tooltipLabel={tooltipLabel}
      format={COMPACT_FORMAT[metric]}
      tooltipFormat={EXACT_FORMAT[metric]}
      totalFor={(key) => totals.get(key)}
      showTotals
      yAxisWidth={Y_AXIS_WIDTH}
      ariaLabel={ariaLabel}
      className={className}
    />
  );
}
