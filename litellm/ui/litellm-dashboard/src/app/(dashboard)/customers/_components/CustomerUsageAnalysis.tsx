import { useMemo, type ReactNode } from "react";
import { parseAsInteger, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData } from "@/components/UsagePage/types";
import { modelColorMap, overviewTotals } from "../../usage/_components/components/overview/overviewData";
import CustomerUsageTrend from "./CustomerUsageTrend";
import CustomerUsageRankings, { CUSTOMER_USAGE_DIMENSIONS } from "./CustomerUsageRankings";
import { rankCustomerModels, rankCustomerUsage } from "./customerUsageData";
import type { CustomerUsageAnalysisData } from "./customerUsageApi";

const USAGE_METRICS = ["spend", "tokens", "cache", "requests"] as const;
export type CustomerUsageMetric = (typeof USAGE_METRICS)[number];
export const customerUsageAnalysisParsers = {
  metric: parseAsStringLiteral(USAGE_METRICS).withDefault("spend"),
  breakdown: parseAsStringLiteral(CUSTOMER_USAGE_DIMENSIONS).withDefault("models"),
  bucket: parseAsStringLiteral(["day", "week"] as const).withDefault("day"),
  search: parseAsString.withDefault(""),
  usagePage: parseAsInteger.withDefault(1),
};
const EMPTY_RESULTS: readonly DailyData[] = [];
const METRIC_LABELS = {
  spend: "customerPortal.analytics.spend",
  tokens: "customerPortal.analytics.tokens",
  requests: "customerPortal.analytics.requests",
} as const;
const TOP_DIMENSION_LABELS = {
  models: "customerPortal.analytics.topModels",
  api_keys: "customerPortal.analytics.topKeys",
  teams: "customerPortal.analytics.topTeams",
} as const;

export default function CustomerUsageAnalysis({
  data,
  loading,
  appFilter,
  children,
  onDrillDown,
}: {
  data: CustomerUsageAnalysisData | undefined;
  loading: boolean;
  appFilter?: ReactNode;
  children?: ReactNode;
  onDrillDown?: (dimension: "teams", id: string) => void;
}) {
  const { t } = useTranslation();
  const [params, setParams] = useQueryStates(customerUsageAnalysisParsers);
  const results = data?.results ?? EMPTY_RESULTS;
  const totals = useMemo(() => overviewTotals(data?.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA, null), [data]);
  const modelColors = useMemo(() => modelColorMap(results), [results]);
  const rankingMetric = params.metric === "cache" ? "tokens" : params.metric;
  const rows = useMemo(
    () =>
      params.breakdown === "models"
        ? rankCustomerModels(results, rankingMetric, totals)
        : rankCustomerUsage(data?.[params.breakdown] ?? [], rankingMetric, totals),
    [data, results, params.breakdown, rankingMetric, totals],
  );
  return (
    <>
      <CustomerUsageTrend
        data={data}
        dimension={params.breakdown}
        onDimensionChange={(breakdown) => void setParams({ breakdown, search: null, usagePage: null })}
        totals={totals}
        metric={params.metric}
        bucket={params.bucket}
        loading={loading}
        colorByModel={modelColors}
        onMetricChange={(metric) => void setParams({ metric, usagePage: null })}
        onBucketChange={(bucket) => void setParams({ bucket })}
        appFilter={appFilter}
      />
      {children}
      {!loading && (
        <>
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-base font-medium tracking-tight">{t(TOP_DIMENSION_LABELS[params.breakdown])}</h2>
            <p className="text-xs text-muted-foreground">
              {t("customerPortal.analytics.sortedBy", { metric: t(METRIC_LABELS[rankingMetric]).toLocaleLowerCase() })}
            </p>
          </div>
          <CustomerUsageRankings
            rows={rows}
            dimension={params.breakdown}
            metric={rankingMetric}
            search={params.search}
            page={params.usagePage}
            colorByModel={modelColors}
            onSearchChange={(search) => void setParams({ search, usagePage: null })}
            onPageChange={(usagePage) => void setParams({ usagePage })}
            onDrillDown={onDrillDown}
          />
        </>
      )}
    </>
  );
}
