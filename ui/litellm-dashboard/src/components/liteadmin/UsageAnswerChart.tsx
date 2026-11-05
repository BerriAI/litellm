import { BarChart } from "@/components/shared/charts/bar_chart";
import { LineChart } from "@/components/shared/charts/line_chart";
import { STACKED_USAGE_PALETTE } from "@/components/shared/charts/StackedUsageChart";
import { formatUsageChartValue, type UsageChart } from "./usageChart";

export function UsageAnswerChart({ chart }: { chart: UsageChart }) {
  const formatValue = (value: number) => formatUsageChartValue(value, chart.unit);
  const unavailable = chart.data.filter((point) => point.value == null).map((point) => point.label);
  const data = chart.data.map((point) => ({ label: point.label, [chart.unit]: point.value }));
  return (
    <figure aria-label={chart.title} className="space-y-3 rounded-xl border bg-card p-4">
      <figcaption className="space-y-1">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm font-medium">{chart.title}</p>
          <p className="text-xl font-semibold tracking-tight tabular-nums">
            {formatUsageChartValue(chart.total, chart.unit)}
            {chart.unit !== "usd" && (
              <span className="ml-1.5 text-sm font-normal text-muted-foreground">{chart.unit}</span>
            )}
          </p>
        </div>
        <p className="text-xs text-muted-foreground">
          {chart.scope} · {chart.period}
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
      <table className="sr-only" aria-label={`${chart.title} values`}>
        <thead>
          <tr>
            <th scope="col">Category</th>
            <th scope="col">{chart.unit}</th>
          </tr>
        </thead>
        <tbody>
          {chart.data.map((point, index) => (
            <tr key={`${point.label}-${index}`}>
              <th scope="row">{point.label}</th>
              <td>{formatUsageChartValue(point.value, chart.unit)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {(chart.note || unavailable.length > 0) && (
        <p className="text-xs leading-relaxed text-muted-foreground">
          {chart.note}
          {unavailable.length > 0 && ` Values unavailable for: ${unavailable.join(", ")}.`}
        </p>
      )}
    </figure>
  );
}
