"use client";

import React from "react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { ArrowDownRight, ArrowUpRight, BarChart3, Minus } from "lucide-react";

import { apiClient } from "@/components/networking";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { PageHeader } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";

type Metric = "requests" | "spend" | "tokens";
type ModelMetric = {
  model_group: string;
  model: string;
  provider: string;
  spend: number;
  prompt_tokens: number;
  completion_tokens: number;
  requests: number;
  successful_requests: number;
  failed_requests: number;
};
type DailyMetric = ModelMetric & { date: string };
type TaskMetric = ModelMetric & { task_type: string };
type ModelInsightsResponse = {
  start_date: string;
  end_date: string;
  daily: DailyMetric[];
  top_models: ModelMetric[];
  by_task: TaskMetric[];
};

const COLORS = ["#2563eb", "#7c3aed", "#0d9488", "#ea580c", "#db2777", "#65a30d"];

const metricValue = (row: ModelMetric, metric: Metric) => {
  if (metric === "requests") return row.requests;
  if (metric === "spend") return row.spend;
  return row.prompt_tokens + row.completion_tokens;
};

const formatMetric = (value: number, metric: Metric) => {
  if (metric === "spend") return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
};

const buildChartData = (rows: DailyMetric[], models: string[], metric: Metric) => {
  const dates = [...new Set(rows.map((row) => row.date))];
  return dates.map((date) => ({
    date,
    ...Object.fromEntries(
      models.map((model) => [
        model,
        rows
          .filter((row) => row.date === date && row.model_group === model)
          .reduce((sum, row) => sum + metricValue(row, metric), 0),
      ]),
    ),
  }));
};

const topChartModels = (rows: DailyMetric[], models: string[], metric: Metric) => {
  const totals = Object.fromEntries(
    models.map((model) => [
      model,
      rows.filter((row) => row.model_group === model).reduce((sum, row) => sum + metricValue(row, metric), 0),
    ]),
  );
  return [...models].sort((left, right) => totals[right] - totals[left]).slice(0, 6);
};

const modelTrend = (rows: DailyMetric[], model: string) => {
  const dates = [...new Set(rows.map((row) => row.date))];
  const midpoint = Math.max(1, Math.floor(dates.length / 2));
  const total = (selected: string[]) =>
    rows
      .filter((row) => row.model_group === model && selected.includes(row.date))
      .reduce((sum, row) => sum + row.prompt_tokens + row.completion_tokens, 0);
  const previous = total(dates.slice(0, midpoint));
  const current = total(dates.slice(midpoint));
  return previous === 0 ? 0 : ((current - previous) / previous) * 100;
};

const groupTasks = (rows: TaskMetric[]) => {
  const tasks = [...new Set(rows.map((row) => row.task_type))];
  return tasks.map((task) => [task, rows.filter((row) => row.task_type === task)] as const);
};

const trendIcon = (value: number) => {
  if (value > 0.5) return <ArrowUpRight className="size-3.5" />;
  if (value < -0.5) return <ArrowDownRight className="size-3.5" />;
  return <Minus className="size-3.5" />;
};

const trendTone = (value: number) => {
  if (value > 0.5) return "text-emerald-600";
  if (value < -0.5) return "text-red-600";
  return "text-muted-foreground";
};

const Trend = ({ value }: { value: number }) => {
  const tone = trendTone(value);
  return (
    <span className={`flex items-center gap-1 text-xs font-medium ${tone}`}>
      {trendIcon(value)} {Math.abs(value).toFixed(1)}%
    </span>
  );
};

export default function ModelInsightsView({ accessToken }: { accessToken: string | null }) {
  const [data, setData] = React.useState<ModelInsightsResponse | null>(null);
  const [metric, setMetric] = React.useState<Metric>("requests");

  React.useEffect(() => {
    if (!accessToken) return;
    void apiClient.get<ModelInsightsResponse>("/model-insights", { accessToken }).then(setData);
  }, [accessToken]);

  if (!data) {
    return (
      <div className="space-y-6 p-8">
        <Skeleton className="h-16 w-96" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  const models = topChartModels(data.daily, [...new Set(data.top_models.map((row) => row.model_group))], metric);
  const chartData = buildChartData(data.daily, models, metric);
  const chartConfig = Object.fromEntries(
    models.map((model, index) => [model, { label: model, color: COLORS[index % COLORS.length] }]),
  ) satisfies ChartConfig;
  const taskGroups = groupTasks(data.by_task);

  return (
    <main className="w-full space-y-6 p-8">
      <PageHeader
        icon={<BarChart3 />}
        title="Model Leaderboard"
        subtitle={`See which models your gateway used from ${data.start_date} through ${data.end_date}`}
      />

      <Card>
        <CardHeader className="flex-row items-start justify-between space-y-0">
          <div>
            <CardTitle>Top model usage</CardTitle>
            <CardDescription>Daily deployment-wide usage, ranked by the metric you select</CardDescription>
          </div>
          <div className="flex rounded-lg border bg-muted/40 p-1">
            {(["requests", "spend", "tokens"] as const).map((value) => (
              <Button
                key={value}
                size="sm"
                variant={metric === value ? "secondary" : "ghost"}
                onClick={() => setMetric(value)}
                className="capitalize"
              >
                {value}
              </Button>
            ))}
          </div>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[360px] w-full aspect-auto">
            <AreaChart data={chartData} margin={{ left: 8, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="date" tickLine={false} axisLine={false} tickFormatter={(value) => value.slice(5)} />
              <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => formatMetric(Number(value), metric)} />
              <ChartTooltip content={<ChartTooltipContent />} />
              {models.map((model, index) => (
                <Area
                  key={model}
                  dataKey={model}
                  type="monotone"
                  stackId="usage"
                  fill={COLORS[index % COLORS.length]}
                  stroke={COLORS[index % COLORS.length]}
                  fillOpacity={0.75}
                />
              ))}
            </AreaChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Top models</CardTitle>
          <CardDescription>
            Ranked by total tokens, with change between the first and second half of the selected period
          </CardDescription>
        </CardHeader>
        <CardContent className="divide-y">
          {data.top_models.map((row, index) => (
            <div
              key={`${row.model_group}-${row.provider}`}
              className="grid grid-cols-[2rem_1fr_auto_auto] items-center gap-4 py-3"
            >
              <span className="text-sm tabular-nums text-muted-foreground">{index + 1}</span>
              <div className="flex min-w-0 items-center gap-3">
                <ProviderLogo provider={row.provider} className="size-7" />
                <div className="min-w-0">
                  <p className="truncate font-medium">{row.model_group}</p>
                  <p className="truncate text-xs text-muted-foreground">{row.model}</p>
                </div>
              </div>
              <span className="text-sm font-medium tabular-nums">
                {formatMetric(row.prompt_tokens + row.completion_tokens, "tokens")} tokens
              </span>
              <Trend value={modelTrend(data.daily, row.model_group)} />
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Top models by task</CardTitle>
          <CardDescription>
            Actual gateway usage grouped by API workload, without inspecting prompt content
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {taskGroups.map(([task, rows]) => {
            const ordered = [...rows].sort((a, b) => metricValue(b, "tokens") - metricValue(a, "tokens")).slice(0, 3);
            return (
              <div key={task} className="rounded-xl border bg-muted/20 p-4">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="font-semibold capitalize">{task}</h3>
                  <Badge variant="secondary">
                    {formatMetric(
                      ordered.reduce((sum, row) => sum + metricValue(row, "tokens"), 0),
                      "tokens",
                    )}{" "}
                    tokens
                  </Badge>
                </div>
                <div className="space-y-3">
                  {ordered.map((row) => (
                    <div key={`${task}-${row.model_group}`} className="flex items-center gap-2">
                      <ProviderLogo provider={row.provider} className="size-5" />
                      <span className="min-w-0 flex-1 truncate text-sm">{row.model_group}</span>
                      <span className="text-xs tabular-nums text-muted-foreground">
                        {formatMetric(metricValue(row, "tokens"), "tokens")}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Cost per session</CardTitle>
          <CardDescription>Session cost is not estimated from request counts</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Add a stable session_id to requests to unlock accurate session-level model comparisons in a future bounded
            session rollup
          </p>
        </CardContent>
      </Card>
    </main>
  );
}
