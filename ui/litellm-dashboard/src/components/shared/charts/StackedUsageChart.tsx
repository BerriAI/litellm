"use client";

import * as React from "react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { type ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { cn } from "@/lib/cva.config";

/** The Model Leaderboard palette; shared so the usage page stacks in the same colors. */
export const STACKED_USAGE_PALETTE = [
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
] as const;

export const stackedUsageColor = (index: number) => STACKED_USAGE_PALETTE[index % STACKED_USAGE_PALETTE.length];

export type StackedUsageScale = "linear" | "log";

export interface StackedUsageChartProps {
  data: readonly Record<string, unknown>[];
  /** Series keys, bottom of the stack first. */
  series: readonly string[];
  /** Display names for `series`, same order; defaults to the keys. */
  labels?: readonly string[];
  /** Overrides the palette per series, e.g. to grey out an "Other" bucket. */
  colors?: readonly string[];
  xKey: string;
  /** Turns an `xKey` value into its tick and tooltip text, e.g. an ISO date into "Oct 3". */
  xLabel?: (value: string) => string;
  format: (value: number) => string;
  /** Appended to the tooltip header as "· Total …" for the hovered bucket, looked up by its `xKey` value. */
  totalFor?: (xValue: string) => number | undefined;
  totalLabel?: string;
  scale?: StackedUsageScale;
  className?: string;
}

/**
 * One tooltip row: the default swatch layout, but the value goes through `format` and is held apart
 * from long series names. Zero rows render nothing, so a day lists only what it actually used.
 */
export function StackedUsageTooltipRow({
  value,
  name,
  color,
  format,
}: {
  value: number;
  name: string;
  color: string | undefined;
  format: (value: number) => string;
}) {
  if (value === 0) return null;
  return (
    <div className="flex w-full items-center gap-2">
      <span className="size-2.5 shrink-0 rounded-[2px]" style={{ backgroundColor: color }} />
      <span className="min-w-0 flex-1 truncate text-muted-foreground">{name}</span>
      <span className="ml-3 font-medium tabular-nums text-foreground">{format(value)}</span>
    </div>
  );
}

/** Daily usage stacked by series, with a tooltip listing each series and the bucket total. */
export function StackedUsageChart({
  data,
  series,
  labels,
  colors,
  xKey,
  xLabel = String,
  format,
  totalFor,
  totalLabel = "Total",
  scale = "linear",
  className,
}: StackedUsageChartProps) {
  const fill = (index: number) => colors?.[index] ?? stackedUsageColor(index);
  const nameOf = (index: number) => labels?.[index] ?? series[index];
  const config = Object.fromEntries(
    series.map((key, index) => [key, { label: nameOf(index), color: fill(index) }]),
  ) satisfies ChartConfig;
  const tooltipHeader = (_label: unknown, payload: readonly { payload?: Record<string, unknown> }[]) => {
    const xValue = String(payload?.[0]?.payload?.[xKey] ?? "");
    const total = totalFor?.(xValue);
    return total === undefined ? xLabel(xValue) : `${xLabel(xValue)} · ${totalLabel} ${format(total)}`;
  };

  return (
    <ChartContainer config={config} className={cn("aspect-auto h-[380px] w-full", className)}>
      <BarChart data={[...data]} margin={{ left: 8, right: 8 }} barCategoryGap="15%" maxBarSize={64}>
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey={xKey}
          tickLine={false}
          axisLine={false}
          minTickGap={48}
          tickFormatter={(value) => xLabel(String(value))}
        />
        <YAxis
          scale={scale}
          domain={scale === "log" ? [1, "auto"] : [0, "auto"]}
          allowDataOverflow
          tickLine={false}
          axisLine={false}
          tickFormatter={(value) => format(Number(value))}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              className="min-w-48"
              labelFormatter={tooltipHeader}
              formatter={(value, _name, item) => (
                <StackedUsageTooltipRow
                  value={Number(value)}
                  name={nameOf(series.indexOf(String(item.dataKey)))}
                  color={item.color}
                  format={format}
                />
              )}
            />
          }
        />
        {series.map((key, index) => (
          <Bar
            key={key}
            dataKey={key}
            name={nameOf(index)}
            stackId="usage"
            fill={fill(index)}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ChartContainer>
  );
}
