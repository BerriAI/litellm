"use client";

import React, { useEffect, useMemo, useState } from "react";

import { AreaChart, BarChart, CustomLegend, DonutChart, SEQUENTIAL_COLOR_RAMP } from "@/components/shared/charts";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { getToolSpend, ToolSpendResponse } from "@/components/networking";
import {
  autorouterOf,
  buildDailyToolSeries,
  cachingOf,
  compressionOf,
  formatRangeLabel,
  localIsoDay,
  MAX_POINTS_WITH_DOTS,
  SAVINGS_COLORS,
  SAVINGS_DRIVERS,
  SAVINGS_SERIES,
  SavingsAccumulation,
  SavingsPoint,
  shortDate,
  toCumulative,
  topToolsBySpend,
  usd,
  withStartAnchor,
} from "./costOptimizationUtils";
import SavingsTiles, { useSavingsTotals } from "@/components/shared/SavingsTiles";
import { DailyActivityRange } from "./useDailyActivityRange";

interface UsageTabProps {
  accessToken: string | null;
  activity: DailyActivityRange;
}

const EMPTY_TOOL_SPEND: ToolSpendResponse = {
  by_tool: [],
  daily: [],
  start_date: null,
  end_date: null,
};

const isoDay = (d: Date): string => d.toISOString().slice(0, 10);

const UsageTab: React.FC<UsageTabProps> = ({ accessToken, activity }) => {
  const { dateValue, onDateChange, results, loading, isFetchingMore } = activity;

  const startTime = dateValue.from ?? null;
  const endTime = dateValue.to ?? null;

  const canViewProxyWideCostData = useCan("viewProxyWideCostData");
  const toolSpendEnabled = canViewProxyWideCostData && !!accessToken && !!startTime && !!endTime;
  const rangeKey = startTime && endTime ? `${isoDay(startTime)}|${isoDay(endTime)}` : "";
  const [toolSpendState, setToolSpendState] = useState<{ key: string; data: ToolSpendResponse } | null>(null);

  useEffect(() => {
    if (!canViewProxyWideCostData || !accessToken || !startTime || !endTime) return;
    let cancelled = false;
    getToolSpend(accessToken, isoDay(startTime), isoDay(endTime))
      .then((res) => {
        if (!cancelled) setToolSpendState({ key: rangeKey, data: res });
      })
      .catch(() => {
        if (!cancelled) setToolSpendState({ key: rangeKey, data: EMPTY_TOOL_SPEND });
      });
    return () => {
      cancelled = true;
    };
  }, [canViewProxyWideCostData, accessToken, startTime, endTime, rangeKey]);

  const toolSpend = toolSpendState?.key === rangeKey ? toolSpendState.data : null;
  const toolSpendLoading = toolSpendEnabled && toolSpend === null;

  const totals = useSavingsTotals(results);

  const [accumulation, setAccumulation] = useState<SavingsAccumulation>("cumulative");

  // The daily rollup arrives newest first; sort on the raw ISO date so the axis
  // reads oldest to newest and the running total accumulates forward in time
  // rather than backward. Sort here, before shortDate() drops the year and makes
  // the labels unsortable.
  const perInterval = useMemo<SavingsPoint[]>(
    () =>
      [...results]
        .sort((a, b) => a.date.localeCompare(b.date))
        .map((d) => ({
          date: shortDate(d.date),
          Compression: compressionOf(d.metrics),
          "Prompt caching": cachingOf(d.metrics),
          "Auto-router": autorouterOf(d.metrics),
        })),
    [results],
  );

  // Cumulative anchors on a synthetic $0 point at the range start so a short
  // range (down to a single day) rises from zero instead of floating as one dot.
  const overTime = useMemo(() => {
    if (accumulation !== "cumulative") return perInterval;
    const startLabel = startTime ? shortDate(localIsoDay(startTime)) : "";
    return withStartAnchor(toCumulative(perInterval), startLabel);
  }, [accumulation, perInterval, startTime]);

  const intervalLabel = "Per day";
  const rangeLabel = formatRangeLabel(startTime ?? undefined, endTime ?? undefined);
  const savingsSubtitle = [
    accumulation === "cumulative" ? "Running total saved" : `Saved ${intervalLabel.toLowerCase()}`,
    rangeLabel && `${rangeLabel} (UTC)`,
  ]
    .filter(Boolean)
    .join(" \u00b7 ");

  // A driver can come out negative (auto-router pays a cold-cache write on every
  // model switch), and a negative slice has no meaning in a donut, so only drivers
  // that actually saved are plotted; the range total keeps the signed truth.
  const byDriver = useMemo(
    () =>
      SAVINGS_DRIVERS.map(({ name, color }) => ({
        driver: name,
        color,
        usd: { Compression: totals.compression, "Prompt caching": totals.caching, "Auto-router": totals.autorouter }[
          name
        ],
      })).filter((d) => d.usd > 0),
    [totals],
  );
  const plottedDriverTotal = useMemo(() => byDriver.reduce((sum, d) => sum + d.usd, 0), [byDriver]);

  const topTools = useMemo(() => topToolsBySpend(toolSpend?.by_tool ?? []), [toolSpend]);
  const topToolNames = useMemo(() => topTools.map((t) => t.tool_name), [topTools]);
  const topToolsChart = useMemo<Record<string, string | number>[]>(
    () => topTools.map((t) => ({ tool_name: t.tool_name, spend: t.spend })),
    [topTools],
  );
  const dailyToolSeries = useMemo(
    () =>
      buildDailyToolSeries(toolSpend?.daily ?? [], topToolNames).map((point) => ({
        ...point,
        date: shortDate(String(point.date)),
      })),
    [toolSpend, topToolNames],
  );
  const toolColors = useMemo(() => SEQUENTIAL_COLOR_RAMP.slice(0, Math.max(topToolNames.length, 1)), [topToolNames]);

  return (
    <div className="w-full space-y-6">
      <div className="flex flex-wrap items-center justify-end gap-4">
        <span className="text-sm text-muted-foreground">Spend is bucketed by UTC day</span>
        <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
      </div>

      <SavingsTiles results={results} isLoading={loading || isFetchingMore} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          {/* CardHeader's own slots rather than hand-rolled rows: the action column is
              sized to its content and the title column takes the rest, so the subtitle
              never competes with the controls for width and neither moves when it grows.
              The controls wrap within their column instead of pushing past the card */}
          <CardHeader>
            <CardTitle>Savings</CardTitle>
            <CardDescription>{savingsSubtitle}</CardDescription>
            <CardAction className="flex flex-wrap items-center justify-end gap-x-4 gap-y-2">
              <CustomLegend categories={SAVINGS_SERIES} colors={SAVINGS_COLORS} />
              <Tabs value={accumulation} onValueChange={(value) => setAccumulation(value as SavingsAccumulation)}>
                <TabsList>
                  <TabsTrigger value="cumulative">Cumulative</TabsTrigger>
                  <TabsTrigger value="per-interval">{intervalLabel}</TabsTrigger>
                </TabsList>
              </Tabs>
            </CardAction>
          </CardHeader>
          <CardContent>
            {accumulation === "cumulative" ? (
              <AreaChart
                data={overTime}
                index="date"
                categories={SAVINGS_SERIES}
                colors={SAVINGS_COLORS}
                valueFormatter={usd}
                showLegend={false}
                showDots={overTime.length <= MAX_POINTS_WITH_DOTS}
              />
            ) : (
              // Not stacked: a driver can be negative once a model switch is charged
              // for its cold cache, and stacking would draw that segment below the axis
              // while the remaining bar still read as the day's total
              <BarChart
                data={overTime}
                index="date"
                categories={SAVINGS_SERIES}
                colors={SAVINGS_COLORS}
                valueFormatter={usd}
                showLegend={false}
              />
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Savings by driver</CardTitle>
          </CardHeader>
          <CardContent>
            <DonutChart
              className="h-80"
              data={byDriver}
              index="driver"
              category="usd"
              colors={byDriver.map((d) => d.color)}
              valueFormatter={usd}
              showLabel
              label={usd(plottedDriverTotal)}
            />
          </CardContent>
        </Card>
      </div>

      {canViewProxyWideCostData && (
        <Card>
          <CardHeader>
            <CardTitle>Spend by tool</CardTitle>
            <p className="text-sm text-muted-foreground">
              Spend on requests that invoked each tool (MCP and client-side tools); declaring a tool without invoking it
              does not count. A request that invoked multiple tools counts its full spend toward each, so this
              attributes rather than partitions spend.
            </p>
          </CardHeader>
          <CardContent>
            {topTools.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                {toolSpendLoading ? "Loading..." : "No tool usage in this range."}
              </p>
            ) : (
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div>
                  <p className="mb-2 text-sm font-medium text-muted-foreground">Total by tool</p>
                  <BarChart
                    data={topToolsChart}
                    index="tool_name"
                    categories={["spend"]}
                    colors={toolColors}
                    colorByDatum
                    layout="vertical"
                    yAxisWidth={140}
                    maxBarSize={64}
                    showLegend={false}
                    valueFormatter={usd}
                  />
                </div>
                <div>
                  <p className="mb-2 text-sm font-medium text-muted-foreground">Daily spend by tool</p>
                  <CustomLegend categories={topToolNames} colors={toolColors} />
                  <BarChart
                    data={dailyToolSeries}
                    index="date"
                    categories={topToolNames}
                    colors={toolColors}
                    stack
                    maxBarSize={64}
                    valueFormatter={usd}
                    showLegend={false}
                  />
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default UsageTab;
