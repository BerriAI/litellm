"use client";

import React from "react";
import { Bar, BarChart, CartesianGrid, Treemap, XAxis, YAxis } from "recharts";
import { ArrowDownRight, ArrowUpRight, BarChart3, Layers, Minus } from "lucide-react";

import { apiClient } from "@/components/networking";
import { extractErrorMessage } from "@/utils/errorUtils";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { PageHeader } from "@/components/shared/PageHeader";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  buildWeeklySeries,
  formatMetric,
  Metric,
  ModelInsightsResponse,
  ModelInsightTasksResponse,
  TaskSummary,
  modelOrder,
  rankModels,
  RankedModel,
} from "./modelInsightsData";

const PALETTE = [
  "#ec4899",
  "#a855f7",
  "#f59e0b",
  "#3b82f6",
  "#10b981",
  "#ef4444",
  "#14b8a6",
  "#84cc16",
  "#6366f1",
  "#f97316",
];
const FALLBACK_COLOR = "#64748b";
const CATEGORY_COLORS: Record<string, string> = {
  General: "#ee8650",
  Agent: "#7666e4",
  Code: "#5fb074",
  Data: "#3b82f6",
};
const SCALES = ["linear", "log"] as const;
const METRIC_LABELS: Record<Metric, string> = { requests: "requests", spend: "spend", tokens: "tokens" };
const RANKING_ROWS = 5;

type Scale = (typeof SCALES)[number];

const formatDelta = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(1)}`;

const DeltaBadge = ({ value }: { value: number }) => {
  if (Math.abs(value) < 0.05) {
    return (
      <span className="flex items-center justify-end gap-1 text-xs text-muted-foreground">
        <Minus className="size-3" /> 0.0
      </span>
    );
  }
  const up = value > 0;
  const Icon = up ? ArrowUpRight : ArrowDownRight;
  return (
    <span className={`flex items-center justify-end gap-1 text-xs ${up ? "text-emerald-600" : "text-red-600"}`}>
      <Icon className="size-3" /> {formatDelta(value)}
    </span>
  );
};

const RankingRow = ({ model, rank }: { model: RankedModel; rank: number }) => (
  <li className="grid grid-cols-[1.5rem_2.5rem_1fr_auto] items-center gap-3 py-2">
    <span className="text-sm tabular-nums text-muted-foreground">{rank}</span>
    <ProviderLogo provider={model.provider} className="size-9 rounded-md border p-1" />
    <div className="min-w-0">
      <p className="truncate font-medium">{model.model_group}</p>
      <p className="truncate text-sm text-muted-foreground">by {model.provider}</p>
    </div>
    <div className="text-right">
      <p className="font-medium tabular-nums">{model.share.toFixed(1)}%</p>
      <DeltaBadge value={model.delta} />
    </div>
  </li>
);

type TileProps = TaskSummary & { x: number; y: number; width: number; height: number; index: number };

const TaskTileContent = ({ x, y, width, height, category, label, leader }: TileProps) => {
  if (width <= 0 || height <= 0) return null;
  const color = CATEGORY_COLORS[category] ?? FALLBACK_COLOR;
  const fits = width > 90 && height > 44;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={color} stroke="#fff" strokeWidth={2} />
      {fits && (
        <>
          <text x={x + 12} y={y + 26} fill="#fff" fontSize={16} fontWeight={500}>
            {label}
          </text>
          <text x={x + 12} y={y + 46} fill="#ffffffcc" fontSize={12}>
            {leader}
          </text>
        </>
      )}
    </g>
  );
};

export default function ModelInsightsView({ accessToken }: { accessToken: string | null }) {
  const [loaded, setLoaded] = React.useState<{ metric: Metric; response: ModelInsightsResponse } | null>(null);
  const [metric, setMetric] = React.useState<Metric>("tokens");
  const [scale, setScale] = React.useState<Scale>("linear");
  const [taskMetric, setTaskMetric] = React.useState<Metric>("spend");
  const [taskData, setTaskData] = React.useState<ModelInsightTasksResponse | null>(null);
  const [taskError, setTaskError] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    apiClient
      .get<ModelInsightsResponse>("/model-insights", { accessToken, query: { metric } })
      .then((response) => {
        if (cancelled) return;
        setError(null);
        setLoaded({ metric, response });
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(extractErrorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, metric]);

  React.useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    apiClient
      .get<ModelInsightTasksResponse>("/model-insights/tasks", { accessToken, query: { metric: taskMetric } })
      .then((response) => {
        if (cancelled) return;
        setTaskError(null);
        setTaskData(response);
      })
      .catch((err: unknown) => {
        if (!cancelled) setTaskError(extractErrorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, taskMetric]);

  const data = loaded?.response ?? null;
  const shown = loaded?.metric ?? metric;
  const isStale = loaded !== null && loaded.metric !== metric;
  const range = React.useMemo(() => ({ start: data?.start_date ?? "", end: data?.end_date ?? "" }), [data]);
  const models = React.useMemo(() => (data ? modelOrder(data.daily, shown) : []), [data, shown]);
  const series = React.useMemo(
    () => (data ? buildWeeklySeries(data.daily, models, shown, range) : []),
    [data, models, shown, range],
  );
  const ranking = React.useMemo(
    () => (data ? rankModels(data.top_models, data.daily, shown, range) : []),
    [data, shown, range],
  );
  const tiles = React.useMemo(() => taskData?.tasks ?? [], [taskData]);
  const categoryShares = React.useMemo(
    () =>
      [...new Set(tiles.map((tile) => tile.category))].map((category) => ({
        category,
        share: tiles.filter((tile) => tile.category === category).reduce((sum, tile) => sum + tile.share, 0),
      })),
    [tiles],
  );

  if (error) {
    return (
      <div className="p-8">
        <Alert variant="destructive">
          <AlertTitle>Could not load model insights</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="space-y-6 p-8">
        <Skeleton className="h-16 w-96" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  const chartConfig = Object.fromEntries(
    models.map((model, index) => [model, { label: model, color: PALETTE[index % PALETTE.length] }]),
  ) satisfies ChartConfig;

  return (
    <main className="w-full space-y-6 p-8">
      <PageHeader
        icon={<BarChart3 />}
        title="Model Leaderboard"
        subtitle={`See which models your gateway used from ${data.start_date} through ${data.end_date}`}
      />

      <Card aria-busy={isStale} className={isStale ? "opacity-60 transition-opacity" : "transition-opacity"}>
        <CardHeader className="flex-row items-start justify-between space-y-0">
          <div>
            <CardTitle>Top models</CardTitle>
            <CardDescription>Weekly {METRIC_LABELS[shown]} across your gateway</CardDescription>
          </div>
          <div className="flex items-center gap-3">
            <Tabs value={metric} onValueChange={(value) => setMetric(value as Metric)}>
              <TabsList>
                {(["requests", "spend", "tokens"] as const).map((value) => (
                  <TabsTrigger key={value} value={value} className="capitalize">
                    {value}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
            <Tabs value={scale} onValueChange={(value) => setScale(value as Scale)}>
              <TabsList>
                {SCALES.map((value) => (
                  <TabsTrigger key={value} value={value} className="capitalize">
                    {value}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </div>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[380px] w-full aspect-auto">
            <BarChart data={series} margin={{ left: 8, right: 8 }} barCategoryGap={2}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={48} />
              <YAxis
                scale={scale}
                domain={scale === "log" ? [1, "auto"] : [0, "auto"]}
                allowDataOverflow
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => formatMetric(Number(value), shown)}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              {models.map((model, index) => (
                <Bar
                  key={model}
                  dataKey={model}
                  stackId="usage"
                  fill={PALETTE[index % PALETTE.length]}
                  isAnimationActive={false}
                />
              ))}
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card aria-busy={isStale} className={isStale ? "opacity-60 transition-opacity" : "transition-opacity"}>
        <CardHeader>
          <CardTitle>Leaderboard</CardTitle>
          <CardDescription>
            Share of {METRIC_LABELS[shown]}, with the change between the first and second half of the period
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-x-12 md:grid-cols-2">
          <ol className="divide-y">
            {ranking.slice(0, RANKING_ROWS).map((model, index) => (
              <RankingRow key={model.model_group} model={model} rank={index + 1} />
            ))}
          </ol>
          <ol className="divide-y">
            {ranking.slice(RANKING_ROWS).map((model, index) => (
              <RankingRow key={model.model_group} model={model} rank={RANKING_ROWS + index + 1} />
            ))}
          </ol>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-start justify-between space-y-0">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Layers className="size-5" /> Top models by task
            </CardTitle>
            <CardDescription>
              Each task&apos;s share of {METRIC_LABELS[taskMetric]}, labelled with its leading model
            </CardDescription>
          </div>
          <Select value={taskMetric} onValueChange={(value) => setTaskMetric(value as Metric)}>
            <SelectTrigger className="w-44" aria-label="Task metric">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="spend">Share of spend</SelectItem>
              <SelectItem value="requests">Share of requests</SelectItem>
              <SelectItem value="tokens">Share of tokens</SelectItem>
            </SelectContent>
          </Select>
        </CardHeader>
        <CardContent className="space-y-4">
          {taskError && (
            <Alert variant="destructive">
              <AlertTitle>Could not load tasks</AlertTitle>
              <AlertDescription>{taskError}</AlertDescription>
            </Alert>
          )}
          <ChartContainer config={{}} className="h-[360px] w-full aspect-auto">
            <Treemap
              data={tiles.map((tile) => ({ ...tile, name: tile.task_type }))}
              dataKey="value"
              isAnimationActive={false}
              content={<TaskTileContent {...({} as TileProps)} />}
            />
          </ChartContainer>
          <ul className="flex flex-wrap gap-x-6 gap-y-2">
            {categoryShares.map(({ category, share }) => (
              <li key={category} className="flex items-center gap-2 text-sm">
                <span
                  className="size-3 rounded-full"
                  style={{ backgroundColor: CATEGORY_COLORS[category] ?? FALLBACK_COLOR }}
                />
                <span className="text-muted-foreground">{category}</span>
                <span className="font-medium tabular-nums">{share.toFixed(1)}%</span>
              </li>
            ))}
          </ul>
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
