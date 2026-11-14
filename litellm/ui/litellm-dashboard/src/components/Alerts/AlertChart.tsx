import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, XAxis, YAxis } from "recharts";
import { useTranslation } from "react-i18next";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { AlertPresentation, dateLabel, metricValue } from "./alertPresentation";

export default function AlertChart({ alert, compact = false }: { alert: AlertPresentation; compact?: boolean }) {
  const { t } = useTranslation();
  if (!alert.history?.length)
    return compact ? null : (
      <p className="py-8 text-sm text-muted-foreground">{t("alerts.unified.chart.unavailable")}</p>
    );
  const end = Date.parse(alert.observed);
  const start =
    alert.baseline === undefined
      ? Math.min(...alert.history.map((point) => Date.parse(point.date)))
      : end - 7 * 86400000;
  const values = new Map(alert.history.map((point) => [point.date, point.value]));
  const data = Array.from({ length: Math.min(31, Math.round((end - start) / 86400000) + 1) }, (_, index) => {
    const date = new Date(start + index * 86400000).toISOString().slice(0, 10);
    return { date, value: values.get(date) ?? null };
  });
  return (
    <figure className="min-w-0 space-y-3" aria-label={alert.chartTitle}>
      {!compact && (
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold">{alert.chartTitle}</h2>
          <span className="text-xs text-muted-foreground">
            {dateLabel(data[0].date)} – {dateLabel(alert.observed)} UTC
          </span>
        </div>
      )}
      <ChartContainer
        config={{ value: { label: alert.chartTitle, color: "#2b3fd6" } }}
        className={compact ? "h-20 w-full aspect-auto" : "h-64 w-full aspect-auto"}
      >
        <BarChart accessibilityLayer data={data} margin={{ top: compact ? 2 : 16, right: 8, bottom: 0, left: 0 }}>
          {!compact && <CartesianGrid vertical={false} />}
          <XAxis
            dataKey="date"
            hide={compact}
            axisLine={false}
            tickLine={false}
            tick={{ fill: "var(--muted-foreground)" }}
            minTickGap={16}
            tickFormatter={(date: string) =>
              new Date(`${date}T00:00:00Z`).toLocaleDateString(undefined, {
                month: "short",
                day: "numeric",
                timeZone: "UTC",
              })
            }
          />
          <YAxis
            hide={compact}
            axisLine={false}
            tickLine={false}
            tick={{ fill: "var(--muted-foreground)" }}
            width={62}
            tickFormatter={(value: number) => metricValue(value, alert.unit)}
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                hideIndicator
                labelFormatter={(value) => dateLabel(String(value))}
                formatter={(value) => metricValue(Number(value), alert.unit)}
              />
            }
          />
          {!compact && alert.baseline !== undefined && (
            <ReferenceLine
              y={alert.baseline}
              ifOverflow="extendDomain"
              stroke="var(--muted-foreground)"
              strokeDasharray="4 4"
            />
          )}
          <Bar
            dataKey="value"
            fill="var(--color-value)"
            maxBarSize={compact ? 20 : 52}
            radius={[2, 2, 0, 0]}
            isAnimationActive={false}
          >
            {data.map((point) => (
              <Cell key={point.date} fillOpacity={point.date === alert.observed ? 1 : 0.35} />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
      {!compact && (
        <>
          <figcaption className="text-xs leading-5 text-muted-foreground">
            {alert.baseline !== undefined &&
              t("alerts.unified.chart.baseline", { value: metricValue(alert.baseline, alert.unit) })}
            {t("alerts.unified.chart.missingDays")}
            {alert.baseline !== undefined && t("alerts.unified.chart.currentUsage")}
          </figcaption>
          <table className="sr-only" aria-label={t("alerts.unified.chart.values", { title: alert.chartTitle })}>
            <thead>
              <tr>
                <th>{t("alerts.unified.chart.date")}</th>
                <th>{t("alerts.unified.chart.value")}</th>
              </tr>
            </thead>
            <tbody>
              {data.map((point) => (
                <tr key={point.date}>
                  <td>{point.date}</td>
                  <td>
                    {point.value === null
                      ? t("alerts.migrationDetail.notRecorded")
                      : metricValue(point.value, alert.unit)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </figure>
  );
}
