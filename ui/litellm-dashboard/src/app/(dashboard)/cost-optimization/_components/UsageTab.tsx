"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Collapse } from "antd";

import { AreaChart, BarChart, CustomLegend, DonutChart, DEFAULT_COLOR_CYCLE } from "@/components/shared/charts";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getToolSpend, ToolSpendResponse } from "@/components/networking";
import { SpendMetrics } from "@/components/UsagePage/types";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import {
  buildDailyToolSeries,
  formatHourBucket,
  formatRangeLabel,
  MAX_POINTS_WITH_DOTS,
  SAVINGS_SERIES,
  SavingsAccumulation,
  SavingsPoint,
  spanInDays,
  toCumulative,
  topToolsBySpend,
  usd,
} from "./costOptimizationUtils";
import { DailyActivityRange } from "./useDailyActivityRange";
import { useHourlySavings } from "./useHourlySavings";

interface UsageTabProps {
  accessToken: string | null;
  activity: DailyActivityRange;
}

const EMPTY_TOOL_SPEND: ToolSpendResponse = {
  by_tool: [],
  daily: [],
  total_spend: 0,
  start_date: null,
  end_date: null,
};

const SAVINGS_COLORS = ["emerald", "blue"] as const;

const shortDate = (iso: string): string =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });

const isoDay = (d: Date): string => d.toISOString().slice(0, 10);

const compressionOf = (m: SpendMetrics): number => m.compression_savings_spend ?? 0;
const cachingOf = (m: SpendMetrics): number => m.prompt_caching_savings_spend ?? 0;
const savedTokensOf = (m: SpendMetrics): number => m.compression_saved_tokens ?? 0;

const MethodologyNote = () => (
  <Collapse
    ghost
    items={[
      {
        key: "methodology",
        label: <span className="text-sm font-medium">How savings are calculated</span>,
        children: (
          <div className="space-y-3 text-sm text-muted-foreground">
            <p>
              Savings are computed for each request when it is logged, using the provider&apos;s reported usage and the
              model&apos;s pricing, then summed into a daily rollup. Totals below are read from that rollup over the
              selected date range, so the numbers never require a scan of raw request logs.
            </p>
            <p>
              Compression savings are the tokens Headroom removed before the call, priced at the model&apos;s input
              rate: <code>compression_saved_tokens * input_cost_per_token</code>
            </p>
            <p>
              Prompt caching savings are the tokens the provider served from cache (Anthropic{" "}
              <code>cache_read_input_tokens</code>, or OpenAI-style <code>prompt_tokens_details.cached_tokens</code>),
              priced at the discount between the normal input rate and the cache-read rate:{" "}
              <code>cache_read_input_tokens * max(input_cost_per_token - cache_read_input_token_cost, 0)</code>
            </p>
            <p>
              Total saved is the sum of both drivers. Models without a separate cache-read price in the pricing map
              contribute zero caching savings rather than erroring.
            </p>
          </div>
        ),
      },
    ]}
  />
);

const SummaryCard = ({ label, value, hint }: { label: string; value: string; hint?: string }) => (
  <Card>
    <CardHeader>
      <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
    </CardHeader>
    <CardContent>
      <p className="text-2xl font-semibold text-foreground">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </CardContent>
  </Card>
);

const UsageTab: React.FC<UsageTabProps> = ({ accessToken, activity }) => {
  const { dateValue, onDateChange, results, loading, isFetchingMore, isAdmin } = activity;

  const startTime = dateValue.from ?? null;
  const endTime = dateValue.to ?? null;

  const toolSpendEnabled = !!accessToken && !!startTime && !!endTime;
  const rangeKey = startTime && endTime ? `${isoDay(startTime)}|${isoDay(endTime)}` : "";
  const [toolSpendState, setToolSpendState] = useState<{ key: string; data: ToolSpendResponse } | null>(null);

  useEffect(() => {
    if (!accessToken || !startTime || !endTime) return;
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
  }, [accessToken, startTime, endTime, rangeKey]);

  const toolSpend = toolSpendState?.key === rangeKey ? toolSpendState.data : null;
  const toolSpendLoading = toolSpendEnabled && toolSpend === null;

  const compressionTotal = useMemo(() => results.reduce((sum, d) => sum + compressionOf(d.metrics), 0), [results]);
  const cachingTotal = useMemo(() => results.reduce((sum, d) => sum + cachingOf(d.metrics), 0), [results]);
  const savedTokensTotal = useMemo(() => results.reduce((sum, d) => sum + savedTokensOf(d.metrics), 0), [results]);
  const totalSaved = compressionTotal + cachingTotal;

  const hourly = useHourlySavings(accessToken, startTime ?? undefined, endTime ?? undefined, isAdmin);

  const [accumulation, setAccumulation] = useState<SavingsAccumulation>("cumulative");

  const perInterval = useMemo<SavingsPoint[]>(() => {
    if (!hourly) {
      return results.map((d) => ({
        date: shortDate(d.date),
        Compression: compressionOf(d.metrics),
        "Prompt caching": cachingOf(d.metrics),
      }));
    }
    const withDate = !!startTime && !!endTime && spanInDays(startTime, endTime) > 1;
    return hourly.buckets.map((b) => ({
      date: formatHourBucket(b.bucket_start, withDate),
      Compression: b.compression_savings_spend,
      "Prompt caching": b.prompt_caching_savings_spend,
    }));
  }, [hourly, results, startTime, endTime]);

  const overTime = useMemo(
    () => (accumulation === "cumulative" ? toCumulative(perInterval) : perInterval),
    [accumulation, perInterval],
  );

  const intervalLabel = hourly ? "Per hour" : "Per day";
  const rangeLabel = formatRangeLabel(startTime ?? undefined, endTime ?? undefined);
  const savingsSubtitle = [
    accumulation === "cumulative" ? "Running total saved" : `Saved ${intervalLabel.toLowerCase()}`,
    rangeLabel,
  ]
    .filter(Boolean)
    .join(" \u00b7 ");

  const byDriver = useMemo(
    () =>
      [
        { driver: "Compression", usd: compressionTotal },
        { driver: "Prompt caching", usd: cachingTotal },
      ].filter((d) => d.usd > 0),
    [compressionTotal, cachingTotal],
  );

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
  const toolColors = useMemo(() => DEFAULT_COLOR_CYCLE.slice(0, Math.max(topToolNames.length, 1)), [topToolNames]);

  return (
    <div className="w-full space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <MethodologyNote />
        <AdvancedDatePicker value={dateValue} onValueChange={onDateChange} />
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <SummaryCard
          label="Total saved"
          value={usd(totalSaved)}
          hint={loading || isFetchingMore ? "Loading..." : "Compression + prompt caching"}
        />
        <SummaryCard
          label="Compression savings"
          value={usd(compressionTotal)}
          hint={`${formatNumberWithCommas(savedTokensTotal)} tokens compressed`}
        />
        <SummaryCard label="Prompt caching savings" value={usd(cachingTotal)} hint="Cache read discount" />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <CardTitle>Savings</CardTitle>
                <p className="text-sm text-muted-foreground">{savingsSubtitle}</p>
              </div>
              <div className="flex items-center gap-4">
                <CustomLegend categories={SAVINGS_SERIES} colors={SAVINGS_COLORS} />
                <Tabs value={accumulation} onValueChange={(value) => setAccumulation(value as SavingsAccumulation)}>
                  <TabsList>
                    <TabsTrigger value="cumulative">Cumulative</TabsTrigger>
                    <TabsTrigger value="per-interval">{intervalLabel}</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <AreaChart
              data={overTime}
              index="date"
              categories={SAVINGS_SERIES}
              colors={SAVINGS_COLORS}
              valueFormatter={usd}
              showLegend={false}
              showDots={overTime.length <= MAX_POINTS_WITH_DOTS}
            />
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
              colors={["emerald", "blue"]}
              valueFormatter={usd}
              showLabel
              label={usd(totalSaved)}
            />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Spend by tool</CardTitle>
          <p className="text-sm text-muted-foreground">
            Spend on requests that called each tool (MCP and client-side tools). A request that used multiple tools
            counts its full spend toward each, so this attributes rather than partitions spend.
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
                  colors={["emerald"]}
                  layout="vertical"
                  yAxisWidth={140}
                  showLegend={false}
                  valueFormatter={usd}
                />
              </div>
              <div>
                <p className="mb-2 text-sm font-medium text-muted-foreground">Daily spend by tool</p>
                <BarChart
                  data={dailyToolSeries}
                  index="date"
                  categories={topToolNames}
                  colors={toolColors}
                  stack
                  valueFormatter={usd}
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default UsageTab;
