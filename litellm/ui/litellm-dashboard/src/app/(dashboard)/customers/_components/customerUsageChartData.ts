import {
  rankValue,
  rollUpBreakdown,
  seriesBy,
  type Series,
  type UsageMetric,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { stackedUsageColor } from "@/components/shared/charts";
import type { DailyData } from "@/components/UsagePage/types";
import i18n from "@/i18n/i18n";
import type { CustomerHierarchyUsage } from "./customerUsageApi";
import type { CustomerUsageDimension } from "./CustomerUsageRankings";

const TOP_SERIES = 8;

const entityDays = (
  primary: readonly DailyData[],
  entities: readonly DailyData[],
  labels: ReadonlyMap<string, string>,
): DailyData[] => {
  const byDate = new Map(entities.map((day) => [day.date, day]));
  return primary.map((day) => ({
    ...day,
    breakdown: {
      ...day.breakdown,
      model_groups: Object.fromEntries(
        Object.entries(byDate.get(day.date)?.breakdown.entities ?? {}).filter(([id]) => id && labels.has(id)),
      ),
    },
  }));
};

export const customerUsageSeries = (
  data: CustomerHierarchyUsage,
  dimension: CustomerUsageDimension,
  metric: UsageMetric,
  colorByModel: ReadonlyMap<string, string>,
): Series => {
  if (dimension === "models") {
    const days = data.results.map((day) => ({
      ...day,
      breakdown: {
        ...day.breakdown,
        model_groups:
          Object.keys(day.breakdown.model_groups).length > 0 ? day.breakdown.model_groups : day.breakdown.models,
      },
    }));
    return seriesBy(days, "model_groups", metric, { top: TOP_SERIES, colorByKey: colorByModel });
  }

  const unnamedKey = i18n.t("customerPortal.analytics.unnamedKey");
  const rows = data[dimension];
  const labels = new Map(
    rows
      .filter((row) => row.id)
      .map((row) => {
        const label = row.label.trim();
        return [row.id, dimension === "api_keys" && (!label || label === row.id) ? unnamedKey : label];
      }),
  );
  const days = dimension === "api_keys" ? data.results : entityDays(data.results, data.teamResults, labels);
  const breakdown = dimension === "api_keys" ? "api_keys" : "model_groups";
  const ranking = rollUpBreakdown(days, breakdown).sort((a, b) => b.spend - a.spend || a.key.localeCompare(b.key));
  const colors = new Map(ranking.map((row, index) => [row.key, stackedUsageColor(index)]));
  const series = seriesBy(days, breakdown, metric, { top: TOP_SERIES, colorByKey: colors });
  const namedCount = Math.min(TOP_SERIES, ranking.filter((row) => rankValue(row, metric) > 0).length);
  return {
    ...series,
    labels: series.labels.map((id, index) => (index < namedCount ? labels.get(id) ?? unnamedKey : id)),
  };
};
