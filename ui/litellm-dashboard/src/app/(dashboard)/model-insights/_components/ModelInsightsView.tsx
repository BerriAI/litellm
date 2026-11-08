"use client";

import { Page } from "@/components/shared/Page";
import type { ParseKeys } from "i18next";
import React from "react";
import { useTranslation } from "react-i18next";
import { ArrowDownRight, ArrowUpRight, BarChart3, Minus } from "lucide-react";
import { StackedUsageChart } from "@/components/shared/charts";

import { apiClient } from "@/components/networking";
import { extractErrorMessage } from "@/utils/errorUtils";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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

const SCALES = ["linear", "log"] as const;
const GRANULARITIES = ["day", "week"] as const;
const METRICS = ["requests", "spend", "tokens"] as const;
const GRANULARITY_LABELS: Record<Granularity, ParseKeys> = {
  day: "modelInsights.modelInsightsView.granularity.day",
  week: "modelInsights.modelInsightsView.granularity.week",
};
const METRIC_LABELS: Record<Metric, ParseKeys> = {
  requests: "modelInsights.modelInsightsView.metrics.requests",
  spend: "modelInsights.modelInsightsView.metrics.spend",
  tokens: "modelInsights.modelInsightsView.metrics.tokens",
};
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

const RankingRow = ({ model, rank }: { model: RankedModel; rank: number }) => {
  const { t } = useTranslation();
  return (
    <li className="grid grid-cols-[1.5rem_2.5rem_1fr_auto] items-center gap-3 py-2">
      <span className="text-sm tabular-nums text-muted-foreground">{rank}</span>
      <ProviderLogo provider={model.provider} className="size-9 rounded-md border p-1" />
      <div className="min-w-0">
        <p className="truncate font-medium">{model.model_group}</p>
        <p className="truncate text-sm text-muted-foreground">
          {t("modelInsights.modelInsightsView.byProvider", { provider: model.provider })}
        </p>
      </div>
      <div className="text-right">
        <p className="font-medium tabular-nums">{model.share.toFixed(1)}%</p>
        <DeltaBadge value={model.delta} />
      </div>
    </li>
  );
};

export default function ModelInsightsView({ accessToken }: { accessToken: string | null }) {
  const { t } = useTranslation();
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
          <AlertTitle>{t("modelInsights.modelInsightsView.loadError")}</AlertTitle>
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

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>
          <BarChart3 />
          {t("modelInsights.modelInsightsView.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>
          {t("modelInsights.modelInsightsView.description", { start: data.start_date, end: data.end_date })}
        </PageHeaderDescription>
      </PageHeader>

      <Card aria-busy={isStale} className={isStale ? "opacity-60 transition-opacity" : "transition-opacity"}>
        <CardHeader className="flex-row items-start justify-between space-y-0">
          <div>
            <CardTitle>{t("modelInsights.modelInsightsView.topModels.title")}</CardTitle>
            <CardDescription>
              {t("modelInsights.modelInsightsView.topModels.description", {
                granularity: t(GRANULARITY_LABELS[granularity]),
                metric: t(METRIC_LABELS[shown]),
              })}
            </CardDescription>
          </div>
          <div className="flex items-center gap-3">
            <Tabs value={metric} onValueChange={(value) => setMetric(value as Metric)}>
              <TabsList>
                {METRICS.map((value) => (
                  <TabsTrigger key={value} value={value} className="capitalize">
                    {t(METRIC_LABELS[value])}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
            <Tabs value={granularity} onValueChange={(value) => setGranularity(value as Granularity)}>
              <TabsList aria-label={t("modelInsights.modelInsightsView.bucketSize")}>
                {GRANULARITIES.map((value) => (
                  <TabsTrigger key={value} value={value}>
                    {t(GRANULARITY_LABELS[value])}
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
          <StackedUsageChart
            data={series}
            series={models}
            xKey="date"
            scale={scale}
            format={(value) => formatMetric(value, shown)}
            totalLabel={t("modelInsights.modelInsightsView.gatewayTotal")}
            totalFor={(label) => bucketTotals.get(label) ?? 0}
          />
        </CardContent>
      </Card>

      <Card aria-busy={isStale} className={isStale ? "opacity-60 transition-opacity" : "transition-opacity"}>
        <CardHeader>
          <CardTitle>{t("modelInsights.modelInsightsView.leaderboard.title")}</CardTitle>
          <CardDescription>
            {t("modelInsights.modelInsightsView.leaderboard.description", { metric: t(METRIC_LABELS[shown]) })}
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
          <CardTitle>{t("modelInsights.modelInsightsView.costPerSession.title")}</CardTitle>
          <CardDescription>{t("modelInsights.modelInsightsView.costPerSession.description")}</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{t("modelInsights.modelInsightsView.costPerSession.hint")}</p>
        </CardContent>
      </Card>
    </Page>
  );
}
