import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { formatAlertValue, METRIC_LABELS } from "@/components/UsageAlerts/usageAlertApi";
import { Issue, issueDate, primaryComparison } from "./issueApi";

export function formatHistoryValue(metric: Parameters<typeof formatAlertValue>[0], value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? formatAlertValue(metric, value) : "Not recorded";
}

export function issueHistory(issue: Issue) {
  const comparison = primaryComparison(issue);
  const history = issue.deprecation?.daily_requests ?? comparison?.history ?? [];
  const metric = issue.deprecation ? "requests" : comparison?.metric ?? "requests";
  const start = comparison?.baseline_start ?? issue.range_start;
  const end = issue.observed_on;
  const days = Math.min(31, Math.floor((Date.parse(end) - Date.parse(start)) / 86_400_000) + 1);
  const values = new Map(history.map((point) => [point.day, point.value]));
  const data =
    history.length === 0
      ? []
      : Array.from({ length: Math.max(0, days) }, (_, index) => {
          const day = new Date(Date.parse(start) + index * 86_400_000).toISOString().slice(0, 10);
          return { day, value: values.get(day) ?? null };
        });
  return { data, metric, comparison };
}

export default function IssueHistoryChart({ issue, compact = false }: { issue: Issue; compact?: boolean }) {
  const { data, metric, comparison } = issueHistory(issue);
  const label = issue.deprecation ? "Daily requests" : `Daily ${METRIC_LABELS[metric].toLowerCase()}`;
  if (data.length === 0)
    return compact ? null : (
      <p className="rounded-md border p-5 text-sm text-muted-foreground">
        Daily history was not saved for this alert. Open activity to explore the recorded traffic.
      </p>
    );
  let caption = "Recorded requests per day. Missing days are left blank.";
  if (comparison) {
    const baseline =
      issue.grouping_reason === "provider_cohort"
        ? "Baseline uses the observed traffic mix across comparable keys and models."
        : "Baseline uses recorded days in this period.";
    caption = `Dashed line: usual ${formatAlertValue(metric, comparison.baseline_value)}. ${baseline} Missing days are left blank.`;
  }
  if (compact) caption = `${issueDate(data[0].day)} – ${issueDate(data[data.length - 1].day)}`;
  return (
    <figure className="min-w-0 space-y-3" aria-label={label}>
      {!compact && (
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold">{label}</h2>
          <span className="text-xs text-muted-foreground">
            {issueDate(data[0].day)} – {issueDate(data[data.length - 1].day)} UTC
          </span>
        </div>
      )}
      <ChartContainer
        config={{ value: { label: METRIC_LABELS[metric], color: "#2b3fd6" } }}
        className={compact ? "h-20 w-full aspect-auto" : "h-64 w-full aspect-auto"}
      >
        <BarChart accessibilityLayer data={data} margin={{ top: compact ? 2 : 16, right: 8, bottom: 0, left: 0 }}>
          {!compact && <CartesianGrid vertical={false} />}
          <XAxis
            dataKey="day"
            axisLine={false}
            tickLine={false}
            hide={compact}
            minTickGap={16}
            tickFormatter={(day: string) =>
              new Date(`${day}T00:00:00Z`).toLocaleDateString(undefined, {
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
            width={62}
            tickFormatter={(value: number) => formatAlertValue(metric, value)}
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                hideIndicator
                labelFormatter={(day) => issueDate(String(day))}
                formatter={(value) => formatHistoryValue(metric, value)}
              />
            }
          />
          {!compact && comparison && (
            <ReferenceLine y={comparison.baseline_value} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
          )}
          <Bar
            dataKey="value"
            fill="var(--color-value)"
            maxBarSize={compact ? 20 : 52}
            radius={[2, 2, 0, 0]}
            isAnimationActive={false}
          >
            {data.map((point) => (
              <Cell key={point.day} fillOpacity={point.day === issue.observed_on ? 1 : 0.35} />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
      <figcaption className="text-xs leading-5 text-muted-foreground">{caption}</figcaption>
      {!compact && (
        <table className="sr-only" aria-label={`${label} values`}>
          <thead>
            <tr>
              <th>Date</th>
              <th>{METRIC_LABELS[metric]}</th>
            </tr>
          </thead>
          <tbody>
            {data.map((point) => (
              <tr key={point.day}>
                <td>{point.day}</td>
                <td>{formatHistoryValue(metric, point.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </figure>
  );
}
