"use client";

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  bucketSeries,
  bucketTotals,
  formatMetricValue,
  labelForDate,
  type UsageMetric,
  type Granularity,
  type OverviewTotals,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { ChartSkeleton, Segmented } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { StackedUsageChart } from "@/components/shared/charts";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { CustomerHierarchyUsage } from "./customerUsageApi";
import { customerUsageSeries } from "./customerUsageChartData";
import { CUSTOMER_USAGE_DIMENSIONS, type CustomerUsageDimension } from "./CustomerUsageRankings";
import { ModelMark } from "@/app/(dashboard)/usage/_components/components/overview/BreakdownChart";

const DIMENSION_LABELS = {
  models: "customerPortal.analytics.models",
  api_keys: "customerPortal.analytics.keys",
  teams: "customerPortal.analytics.teams",
} as const;

interface CustomerUsageTrendProps {
  data: CustomerHierarchyUsage | undefined;
  dimension: CustomerUsageDimension;
  onDimensionChange: (dimension: CustomerUsageDimension) => void;
  totals: OverviewTotals;
  metric: UsageMetric;
  bucket: Granularity;
  loading: boolean;
  colorByModel: ReadonlyMap<string, string>;
  onMetricChange: (metric: UsageMetric) => void;
  onBucketChange: (bucket: Granularity) => void;
}

export default function CustomerUsageTrend({
  data,
  dimension,
  onDimensionChange,
  totals,
  metric,
  bucket,
  loading,
  colorByModel,
  onMetricChange,
  onBucketChange,
}: CustomerUsageTrendProps) {
  const { t } = useTranslation();
  const dailySeries = useMemo(
    () =>
      data
        ? customerUsageSeries(data, dimension, metric, colorByModel)
        : { data: [], keys: [], labels: [], colors: [] },
    [data, dimension, metric, colorByModel],
  );
  const series = useMemo(() => bucketSeries(dailySeries, bucket), [dailySeries, bucket]);
  const bucketValues = useMemo(() => bucketTotals(series), [series]);
  const empty = totals.requests === 0 && totals.tokens === 0 && totals.spend === 0;
  return (
    <section aria-label={t("customerPortal.analytics.trend")} className="border-b pb-5">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-medium tracking-tight">{t("customerPortal.analytics.trend")}</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {t("customerPortal.analytics.trendHint", { dimension: t(DIMENSION_LABELS[dimension]).toLocaleLowerCase() })}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Select value={dimension} onValueChange={(value) => onDimensionChange(value as CustomerUsageDimension)}>
            <SelectTrigger size="sm" aria-label={t("customerPortal.analytics.breakDownBy")}>
              <span className="text-muted-foreground">{t("customerPortal.analytics.breakDownBy")}</span>
              <SelectValue>{t(DIMENSION_LABELS[dimension])}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {CUSTOMER_USAGE_DIMENSIONS.map((value) => (
                <SelectItem key={value} value={value}>
                  {t(DIMENSION_LABELS[value])}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Segmented<UsageMetric>
            label={t("customerPortal.analytics.metric")}
            value={metric}
            options={[
              { value: "spend", label: t("customerPortal.analytics.spend") },
              { value: "tokens", label: t("customerPortal.analytics.tokens") },
              { value: "requests", label: t("customerPortal.analytics.requests") },
            ]}
            onChange={(metric) => onMetricChange(metric)}
          />
          <div className="h-5 w-px bg-border" aria-hidden="true" />
          <Segmented
            label={t("customerPortal.analytics.bucket")}
            value={bucket}
            options={[
              { value: "day", label: t("customerPortal.analytics.daily") },
              { value: "week", label: t("customerPortal.analytics.weekly") },
            ]}
            onChange={(bucket) => onBucketChange(bucket)}
          />
        </div>
      </div>
      {loading && (
        <div role="status" aria-label={t("customerPortal.analytics.loading")}>
          <ChartSkeleton className="h-72 sm:h-80" />
        </div>
      )}
      {!loading && empty && (
        <div className="flex h-72 items-center justify-center text-sm text-muted-foreground">
          {t("customerPortal.analytics.empty")}
        </div>
      )}
      {!loading && !empty && (
        <>
          <StackedUsageChart
            data={series.data}
            series={series.keys}
            labels={series.labels}
            colors={series.colors}
            xKey="date"
            xLabel={(date) => labelForDate(series, date)}
            format={(value) => formatMetricValue(value, metric)}
            totalFor={(date) => bucketValues.get(date)}
            className="h-72 sm:h-80"
          />
          <div className="mt-3 flex flex-wrap justify-center gap-x-5 gap-y-2 text-xs text-muted-foreground">
            {series.labels.map((label, index) => (
              <span key={series.keys[index]} className="inline-flex items-center gap-1.5">
                {dimension === "models" ? (
                  <ModelMark model={label} color={series.colors[index]} />
                ) : (
                  <span
                    aria-hidden="true"
                    className="size-2 rounded-sm"
                    style={{ backgroundColor: series.colors[index] }}
                  />
                )}
                {label}
              </span>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
