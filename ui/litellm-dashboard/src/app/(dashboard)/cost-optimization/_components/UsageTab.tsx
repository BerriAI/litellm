"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Info } from "lucide-react";

import { AreaChart, BarChart, CustomLegend, DonutChart, SEQUENTIAL_COLOR_RAMP } from "@/components/shared/charts";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { getToolSpend, ToolSpendResponse } from "@/components/networking";
import { SpendMetrics } from "@/components/UsagePage/types";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { buildDailyToolSeries, topToolsBySpend, usd } from "./costOptimizationUtils";
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

const shortDate = (iso: string): string =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });

const isoDay = (d: Date): string => d.toISOString().slice(0, 10);

const compressionOf = (m: SpendMetrics): number => m.compression_savings_spend ?? 0;
const cachingOf = (m: SpendMetrics): number => m.prompt_caching_savings_spend ?? 0;
const savedTokensOf = (m: SpendMetrics): number => m.compression_saved_tokens ?? 0;

const SummaryCard = ({ label, value, hint, info }: { label: string; value: string; hint?: string; info?: string }) => (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between space-y-0">
      <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
      {info && (
        <Popover>
          <PopoverTrigger
            aria-label={`How ${label.toLowerCase()} is calculated`}
            data-testid={`summary-card-info-${label.toLowerCase().replace(/\s+/g, "-")}`}
            className="cursor-pointer text-muted-foreground hover:text-foreground"
          >
            <Info className="size-3.5" />
          </PopoverTrigger>
          <PopoverContent align="end" className="w-64 text-sm text-muted-foreground">
            {info}
          </PopoverContent>
        </Popover>
      )}
    </CardHeader>
    <CardContent>
      <p className="text-2xl font-semibold text-foreground">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </CardContent>
  </Card>
);

const UsageTab: React.FC<UsageTabProps> = ({ accessToken, activity }) => {
  const { dateValue, onDateChange, results, loading, isFetchingMore } = activity;

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

  const overTime = useMemo(
    () =>
      results.map((d) => ({
        date: shortDate(d.date),
        Compression: compressionOf(d.metrics),
        "Prompt caching": cachingOf(d.metrics),
      })),
    [results],
  );

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
  const toolColors = useMemo(() => SEQUENTIAL_COLOR_RAMP.slice(0, Math.max(topToolNames.length, 1)), [topToolNames]);

  return (
    <div className="w-full space-y-6">
      <div className="flex flex-wrap items-center justify-end gap-4">
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
          info="Tokens Headroom removed before the call, priced at the model's input rate."
        />
        <SummaryCard
          label="Prompt caching savings"
          value={usd(cachingTotal)}
          hint="Cache read discount"
          info="Tokens the provider served from cache, priced at the discount between the input and cache-read rates."
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Savings over time</CardTitle>
          </CardHeader>
          <CardContent>
            <AreaChart
              data={overTime}
              index="date"
              categories={["Compression", "Prompt caching"]}
              colors={["emerald", "blue"]}
              valueFormatter={usd}
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
            Spend on requests that invoked each tool (MCP and client-side tools); declaring a tool without invoking it
            does not count. A request that invoked multiple tools counts its full spend toward each, so this attributes
            rather than partitions spend.
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
    </div>
  );
};

export default UsageTab;
