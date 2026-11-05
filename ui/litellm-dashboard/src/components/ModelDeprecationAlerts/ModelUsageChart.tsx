"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { StackedUsageChart } from "@/components/shared/charts";
import { Button } from "@/components/ui/button";
import { backgroundFetchClient } from "@/lib/http/api";
import { toDailyData } from "@/components/UsagePage/dailyActivityApi";
import { Panel, ChartSkeleton, Segmented } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import {
  bucketTotals,
  formatMetricValue,
  labelForDate,
  seriesBy,
  type UsageMetric,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import type { deprecationUsageRange } from "./deprecationAlerts";

const METRICS = [
  { value: "requests", label: "Requests" },
  { value: "tokens", label: "Tokens" },
  { value: "spend", label: "Spend" },
] as const;

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
  const [metric, setMetric] = useState<UsageMetric>("requests");
  const activity = useQuery({
    queryKey: ["deprecationDailyUsage", scope, model, team, range],
    queryFn: async ({ signal }) => {
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
  });
  const series = useMemo(() => {
    const days = activity.data ? toDailyData(activity.data) : [];
    const aliases = new Map(
      days.flatMap((day) =>
        Object.entries(day.breakdown.api_keys).map(
          ([id, entry]) => [id, entry.metadata.key_alias || `Key · ${id.slice(0, 6)}`] as const,
        ),
      ),
    );
    const result = seriesBy(days, "api_keys", metric, 8);
    result.labels = result.labels.map((label) => aliases.get(label) || label);
    return result;
  }, [activity.data, metric]);
  const totals = useMemo(() => bucketTotals(series), [series]);
  return (
    <Panel
      title="Daily usage by key"
      subtitle={`Last 30 days · ${team ? "Selected team" : "All teams"} · Each color is a key`}
      className="shrink-0"
      action={<Segmented label="Usage metric" value={metric} options={METRICS} onChange={setMetric} />}
    >
      {activity.isPending ? (
        <ChartSkeleton className="h-64" />
      ) : activity.isError ? (
        <div role="alert" className="py-6 text-sm">
          Could not load daily usage.{" "}
          <Button variant="outline" size="sm" onClick={() => void activity.refetch()}>
            Retry
          </Button>
        </div>
      ) : series.data.length === 0 ? (
        <p className="py-10 text-center text-sm text-muted-foreground">No daily usage recorded for this model.</p>
      ) : (
        <>
          <StackedUsageChart
            data={series.data}
            series={series.keys}
            labels={series.labels}
            colors={series.colors}
            xKey="date"
            xLabel={(date) => labelForDate(series, date)}
            format={(value) => formatMetricValue(value, metric)}
            totalFor={(date) => totals.get(date)}
            className="h-64"
          />
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2 text-xs text-muted-foreground">
            {series.labels.map((label, index) => (
              <span key={series.keys[index]} className="inline-flex items-center gap-1.5">
                <span className="size-2 rounded-sm" style={{ backgroundColor: series.colors[index] }} />
                {label}
              </span>
            ))}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Up to eight keys shown separately. Other includes the remaining usage; the table below lists all affected
            keys.
          </p>
        </>
      )}
    </Panel>
  );
}
