"use client";

import { Page } from "@/components/shared/Page";
import React from "react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { ArrowDownRight, ArrowUpRight, BarChart3, Minus } from "lucide-react";

import { apiClient } from "@/components/networking";
import { extractErrorMessage } from "@/utils/errorUtils";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  buildBucketTotals,
  buildSeries,
  formatMetric,
  Granularity,
  Metric,
  ModelInsightsResponse,
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
const SCALES = ["linear", "log"] as const;
const GRANULARITIES = ["day", "week"] as const;
const GRANULARITY_LABELS: Record<Granularity, string> = { day: "Daily", week: "Weekly" };
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

export default function ModelInsightsView({ accessToken }: { accessToken: string | null }) {
  const [loaded, setLoaded] = React.useState<{ metric: Metric; response: ModelInsightsResponse } | null>(null);
  const [metric, setMetric] = React.useState<Metric>("tokens");
  const [scale, setScale] = React.useState<Scale>("linear");
  const [granularity, setGranularity] = React.useState<Granularity>("day");
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

  const data = loaded?.response ?? null;
  const shown = loaded?.metric ?? metric;
  const isStale = loaded !== null && loaded.metric !== metric;
  const range = React.useMemo(() => ({ start: data?.start_date ?? "", end: data?.end_date ?? "" }), [data]);
  const models = React.useMemo(() => (data ? modelOrder(data.daily, shown) : []), [data, shown]);
  const series = React.useMemo(
    () => (data ? buildSeries(data.daily, models, shown, { ...range, granularity }) : []),
    [data, models, shown, range, granularity],
  );
  const bucketTotals = React.useMemo(
    () => (data ? buildBucketTotals(data.daily_totals, shown, { ...range, granularity }) : new Map<string, number>()),
    [data, shown, range, granularity],
  );
  const ranking = React.useMemo(
    () => (data ? rankModels(data.top_models, data.daily, shown, range) : []),
    [data, shown, range],
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
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <BarChart3 />
          Model Leaderboard
        </PageHeaderTitle>
        <PageHeaderDescription>
          See which models your gateway used from {data.start_date} through {data.end_date}
        </PageHeaderDescription>
      </PageHeader>

      <Card aria-busy={isStale} className={isStale ? "opacity-60 transition-opacity" : "transition-opacity"}>
        <CardHeader className="flex-row items-start justify-between space-y-0">
          <div>
            <CardTitle>Top models</CardTitle>
            <CardDescription>
              {GRANULARITY_LABELS[granularity]} {METRIC_LABELS[shown]} across your gateway
            </CardDescription>
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
            <Tabs value={granularity} onValueChange={(value) => setGranularity(value as Granularity)}>
              <TabsList aria-label="Bucket size">
                {GRANULARITIES.map((value) => (
                  <TabsTrigger key={value} value={value}>
                    {GRANULARITY_LABELS[value]}
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
            <BarChart data={series} margin={{ left: 8, right: 8 }} barCategoryGap="15%" maxBarSize={64}>
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
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(label) =>
                      `${label} · Gateway total ${formatMetric(bucketTotals.get(String(label)) ?? 0, shown)}`
                    }
                  />
                }
              />
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
    </Page>
  );
}
