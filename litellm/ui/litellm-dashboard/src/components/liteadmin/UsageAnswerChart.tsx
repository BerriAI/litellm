import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { BarChart } from "@/components/shared/charts/bar_chart";
import { LineChart } from "@/components/shared/charts/line_chart";
import { STACKED_USAGE_PALETTE } from "@/components/shared/charts/StackedUsageChart";
import { formatUsageChartValue, type UsageChart } from "./usageChart";

const CHART_TITLES: Record<string, ParseKeys> = {
  "Daily spend": "aria.chart.titles.dailySpend",
  "Daily requests": "aria.chart.titles.dailyRequests",
  "Daily tokens": "aria.chart.titles.dailyTokens",
  "Spend by model": "aria.chart.titles.spendByModel",
  "Spend by provider": "aria.chart.titles.spendByProvider",
  "Token breakdown": "aria.chart.titles.tokenBreakdown",
};

const CHART_SCOPES: Record<string, ParseKeys> = {
  "Gateway usage": "aria.chart.scopes.gateway",
  "All teams": "aria.chart.scopes.allTeams",
  "All tags": "aria.chart.scopes.allTags",
  "Your usage": "aria.chart.scopes.yours",
};

const CHART_LABELS: Record<string, ParseKeys> = {
  Unknown: "aria.chart.labels.unknown",
  Other: "aria.chart.labels.other",
  Prompt: "aria.chart.labels.prompt",
  Completion: "aria.chart.labels.completion",
  "Cache read": "aria.chart.labels.cacheRead",
  "Cache creation": "aria.chart.labels.cacheCreation",
};

const CHART_NOTES: Record<string, ParseKeys> = {
  "Up to six leading entries. Other includes remaining recorded spend.": "aria.chart.notes.otherRemainder",
  "Cache counts are reported separately. Do not add them to total tokens.": "aria.chart.notes.cacheSeparate",
};

export function UsageAnswerChart({ chart }: { chart: UsageChart }) {
  const { t } = useTranslation();
  const display = (keys: Record<string, ParseKeys>, text: string) => (Object.hasOwn(keys, text) ? t(keys[text]) : text);
  const title = display(CHART_TITLES, chart.title);
  const formatValue = (value: number) => formatUsageChartValue(value, chart.unit);
  const unavailable = chart.data
    .filter((point) => point.value == null)
    .map((point) => display(CHART_LABELS, point.label));
  const data = chart.data.map((point) => ({ label: display(CHART_LABELS, point.label), [chart.unit]: point.value }));
  return (
    <figure aria-label={title} className="space-y-3 rounded-xl border bg-card p-4">
      <figcaption className="space-y-1">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm font-medium">{title}</p>
          <p className="text-xl font-semibold tracking-tight tabular-nums">
            {formatUsageChartValue(chart.total, chart.unit)}
            {chart.unit !== "usd" && (
              <span className="ml-1.5 text-sm font-normal text-muted-foreground">{chart.unit}</span>
            )}
          </p>
        </div>
        <p className="text-xs text-muted-foreground">
          {display(CHART_SCOPES, chart.scope)} · {chart.period}
        </p>
      </figcaption>
      {chart.kind === "bar" ? (
        <BarChart
          data={data}
          index="label"
          categories={[chart.unit]}
          colors={STACKED_USAGE_PALETTE}
          colorByDatum
          layout="vertical"
          yAxisWidth={160}
          indexFormatter={(label) => (label.length > 22 ? `${label.slice(0, 21)}…` : label)}
          maxBarSize={26}
          showLegend={false}
          valueFormatter={formatValue}
          className={chart.data.length > 4 ? "h-56" : "h-40"}
        />
      ) : (
        <LineChart
          data={data}
          index="label"
          categories={[chart.unit]}
          colors={STACKED_USAGE_PALETTE}
          showDots
          showLegend={false}
          yAxisWidth={80}
          valueFormatter={formatValue}
          className="h-40"
        />
      )}
      <table className="sr-only" aria-label={t("aria.chart.valuesLabel", { title })}>
        <thead>
          <tr>
            <th scope="col">{t("aria.chart.category")}</th>
            <th scope="col">{chart.unit}</th>
          </tr>
        </thead>
        <tbody>
          {chart.data.map((point, index) => (
            <tr key={`${point.label}-${index}`}>
              <th scope="row">{display(CHART_LABELS, point.label)}</th>
              <td>{formatUsageChartValue(point.value, chart.unit)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {(chart.note || unavailable.length > 0) && (
        <p className="text-xs leading-relaxed text-muted-foreground">
          {chart.note && display(CHART_NOTES, chart.note)}
          {unavailable.length > 0 && ` ${t("aria.chart.valuesUnavailable", { labels: unavailable.join(", ") })}`}
        </p>
      )}
    </figure>
  );
}
