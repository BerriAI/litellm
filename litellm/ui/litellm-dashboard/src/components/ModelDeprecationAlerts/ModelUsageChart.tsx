"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { StackedUsageChart } from "@/components/shared/charts";
import { Button } from "@/components/ui/button";
import { backgroundFetchClient } from "@/lib/http/api";
import { toDailyData } from "@/components/UsagePage/dailyActivityApi";
import { Panel, ChartSkeleton } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { seriesBy } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import type { deprecationUsageRange } from "./deprecationAlerts";
import { dailyModelRequests } from "./migrationData";

const formatDate = (date: string) =>
  new Date(`${date}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

export default function ModelUsageChart({
  model,
  team,
  scope,
  range,
}: {
  model: string;
  team: string | null;
  scope: string;
  range: ReturnType<typeof deprecationUsageRange>;
}) {
  const { t } = useTranslation();
  const [legendExpanded, setLegendExpanded] = useState(false);
  const activityOptions = {
    queryKey: ["deprecationDailyUsage", scope, model, team, range],
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const query = { ...range, model, api_key_limit: 100, timezone: 0 };
      if (team) {
        const { data } = await backgroundFetchClient.GET("/team/daily/activity/aggregated", {
          params: { query: { ...query, team_ids: team } },
          signal,
        });
        if (!data) throw new Error("Daily model usage is unavailable");
        return data;
      }
      const { data } = await backgroundFetchClient.GET("/user/daily/activity/aggregated", {
        params: { query },
        signal,
      });
      if (!data) throw new Error("Daily model usage is unavailable");
      return data;
    },
    staleTime: 5 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000,
  };
  const activity = useQuery(activityOptions);
  const daily = useMemo(() => dailyModelRequests(activity.data?.results ?? [], range), [activity.data, range]);
  const totals = useMemo(() => new Map(daily.map((day) => [day.date, day.requests])), [daily]);
  const series = useMemo(() => {
    const days = activity.data ? toDailyData(activity.data) : [];
    const aliases = new Map(
      days.flatMap((day) =>
        Object.entries(day.breakdown.api_keys).map(
          ([id, entry]) =>
            [id, entry.metadata.key_alias || t("alerts.usageChart.keyFallback", { id: id.slice(0, 6) })] as const,
        ),
      ),
    );
    const result = seriesBy(days, "api_keys", "requests", 100);
    const byDate = new Map(result.data.map((day) => [day.date, day]));
    const zeros = Object.fromEntries(result.keys.map((key) => [key, 0]));
    result.data = daily.map((day) => byDate.get(day.date) ?? { date: day.date, label: formatDate(day.date), ...zeros });
    result.labels = result.labels.map((label) => aliases.get(label) || label);
    return result;
  }, [activity.data, daily, t]);
  function renderContent() {
    if (activity.isPending) return <ChartSkeleton className="h-64" />;
    if (activity.isError)
      return (
        <div role="alert" className="py-6 text-sm">
          {t("alerts.usageChart.loadError")}{" "}
          <Button variant="outline" size="sm" onClick={() => void activity.refetch()}>
            {t("alerts.usageChart.retry")}
          </Button>
        </div>
      );
    return (
      <>
        <StackedUsageChart
          data={series.data}
          series={series.keys}
          labels={series.labels}
          colors={series.colors}
          xKey="date"
          xLabel={formatDate}
          format={(value) => value.toLocaleString()}
          totalFor={(date) => totals.get(date)}
          className="h-64"
        />
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-xs text-muted-foreground">
          {series.labels.slice(0, legendExpanded ? undefined : 8).map((label, index) => (
            <span key={series.keys[index]} className="inline-flex items-center gap-1.5">
              <span className="size-2 rounded-sm" style={{ backgroundColor: series.colors[index] }} />
              {label}
            </span>
          ))}
        </div>
        {series.labels.length > 8 && (
          <button
            type="button"
            className="mt-2 self-start text-xs text-primary hover:underline"
            aria-expanded={legendExpanded}
            onClick={() => setLegendExpanded(!legendExpanded)}
          >
            {legendExpanded ? t("alerts.usageChart.showFewer") : t("alerts.usageChart.showAll")}
          </button>
        )}
      </>
    );
  }
  return (
    <Panel
      title={t("alerts.usageChart.title")}
      subtitle={team ? t("alerts.usageChart.subtitleTeam") : t("alerts.usageChart.subtitleAll")}
      className="shrink-0"
    >
      {renderContent()}
      <p className="mt-2 text-xs text-muted-foreground">{t("alerts.usageChart.footnote")}</p>
    </Panel>
  );
}
