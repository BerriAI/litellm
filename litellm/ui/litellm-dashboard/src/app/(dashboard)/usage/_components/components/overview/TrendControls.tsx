"use client";

import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { Segmented } from "./Primitives";
import type { TrendSelectionState } from "./trendSelection";
import type { TrendGranularity, TrendMetric } from "./usageTrendData";

const METRIC_OPTIONS = [
  { value: "tokens", labelKey: "usage.breakdown.metricOptions.tokens" },
  { value: "spend", labelKey: "usage.breakdown.metricOptions.spend" },
  { value: "requests", labelKey: "usage.breakdown.metricOptions.requests" },
] as const satisfies readonly { value: TrendMetric; labelKey: ParseKeys }[];

const GRANULARITY_OPTIONS = [
  { value: "day", labelKey: "usage.breakdownChart.granularityOptions.day" },
  { value: "week", labelKey: "usage.breakdownChart.granularityOptions.week" },
  { value: "month", labelKey: "usage.trend.granularityMonth" },
] as const satisfies readonly { value: TrendGranularity; labelKey: ParseKeys }[];

export default function TrendControls({ state }: { state: TrendSelectionState }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Segmented
        label={t("usage.breakdown.metricLabel")}
        value={state.selection.metric}
        options={METRIC_OPTIONS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }))}
        onChange={state.setMetric}
      />
      <Segmented
        label={t("usage.breakdownChart.bucketSize")}
        value={state.selection.granularity}
        options={GRANULARITY_OPTIONS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }))}
        onChange={state.setGranularity}
      />
    </div>
  );
}
