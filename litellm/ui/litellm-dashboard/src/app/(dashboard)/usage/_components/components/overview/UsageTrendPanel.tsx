"use client";

import { useMemo, type ReactNode } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import type { DailyData } from "@/components/UsagePage/types";
import { cn } from "@/lib/cva.config";
import { modelColorMap } from "./overviewData";
import { ChartSkeleton, Panel, PANEL_INSET_X } from "./Primitives";
import ModelContributionTable from "./ModelContributionTable";
import TrendChart, {
  GRANULARITY_ADJECTIVE,
  METRIC_NOUN,
  PartialPeriodNote,
  TrendLegend,
  TrendNotice,
  type Trend,
} from "./TrendChart";
import { formatExactCount, formatExactUsd } from "./trendFormat";
import type { TrendSelection } from "./trendSelection";
import { formatDayRange, usageTrend, type DayRange, type TrendDimension } from "./usageTrendData";

const LEGEND_HINT: Readonly<Record<TrendDimension, ParseKeys>> = {
  model_groups: "usage.trend.legendHint.modelGroups",
  models: "usage.trend.legendHint.models",
};

interface UsageTrendPanelProps {
  results: readonly DailyData[];
  range: DayRange | null;
  loading: boolean;
  failed: boolean;
  selection: TrendSelection;
  dimension?: TrendDimension;
  colorByModel?: ReadonlyMap<string, string>;
  compact?: boolean;
  action?: ReactNode;
}

function UnattributedNote({ trend }: { trend: Trend }) {
  const { t } = useTranslation();
  const { spend, tokens } = trend.unattributed;
  if (spend === 0 && tokens === 0) return null;
  return (
    <TrendNotice>
      {t("usage.trend.unattributedNote", { spend: formatExactUsd(spend), tokens: formatExactCount(tokens) })}
    </TrendNotice>
  );
}

export default function UsageTrendPanel({
  results,
  range,
  loading,
  failed,
  selection,
  dimension = "model_groups",
  colorByModel: sharedColors,
  compact = false,
  action,
}: UsageTrendPanelProps) {
  const { t } = useTranslation();
  const { metric, granularity } = selection;
  const colorByModel = useMemo(() => sharedColors ?? modelColorMap(results), [sharedColors, results]);
  const trend = useMemo(() => {
    if (range === null) return null;
    const input = { results, range, granularity, colorByModel, dimension, rankBy: metric };
    return usageTrend(input);
  }, [results, range, granularity, colorByModel, dimension, metric]);
  const chartHeight = compact ? "h-56" : "h-[320px]";
  const ready = !loading && trend !== null;
  const hasRows = results.length > 0;
  const showTable = !failed && ready && hasRows;
  const caption = t("usage.trend.caption", {
    granularity: t(GRANULARITY_ADJECTIVE[granularity]),
    metric: t(METRIC_NOUN[metric]),
  });
  const subtitle = range ? t("usage.trend.subtitle", { caption, range: formatDayRange(range) }) : caption;

  const body = () => {
    if (failed) return <TrendNotice>{t("usage.trend.unavailable")}</TrendNotice>;
    if (!ready || trend === null) {
      return (
        <div role="status" aria-label={t("usage.trend.loading")}>
          <ChartSkeleton className={chartHeight} />
        </div>
      );
    }
    if (!hasRows) return <p className="py-10 text-center text-sm text-muted-foreground">{t("usage.trend.empty")}</p>;
    const series = trend.series[metric];
    return (
      <div className="grid gap-4">
        <TrendLegend
          title={t("usage.trend.legend")}
          hint={t(LEGEND_HINT[dimension])}
          labels={series.labels}
          colors={series.colors}
        />
        <TrendChart trend={trend} selection={selection} ariaLabel={caption} className={chartHeight} />
        <PartialPeriodNote trend={trend} granularity={granularity} />
        <UnattributedNote trend={trend} />
      </div>
    );
  };

  return (
    <Panel
      testId="usage-trend"
      title={t("usage.trend.title")}
      subtitle={subtitle}
      action={action}
      bodyClassName="px-0 pt-4 pb-0"
    >
      <div className={PANEL_INSET_X}>{body()}</div>
      {showTable && trend !== null && (
        <section aria-label={t("usage.trend.table.title")} className={cn("mt-4 border-t pt-3 pb-2", PANEL_INSET_X)}>
          <h4 className="text-sm leading-5 font-medium text-foreground">{t("usage.trend.table.title")}</h4>
          <ModelContributionTable rows={trend.rows} total={trend.total} />
        </section>
      )}
    </Panel>
  );
}
