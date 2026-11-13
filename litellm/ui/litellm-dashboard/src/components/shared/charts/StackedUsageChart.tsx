"use client";

import * as React from "react";
import { Bar, BarChart, CartesianGrid, LabelList, XAxis, YAxis } from "recharts";
import { type ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import i18n from "@/i18n/i18n";
import { cn } from "@/lib/cva.config";

const LOG_SCALE_DOMAIN: [number, "auto"] = [1, "auto"];
const LINEAR_SCALE_DOMAIN: [number, "auto"] = [0, "auto"];

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

interface BarTotalLabelProps {
  x?: number | string;
  y?: number | string;
  width?: number | string;
  index?: number;
}

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
  tooltipLabel?: (value: string) => string;
  tooltipFormat?: (value: number) => string;
  showTotals?: boolean;
  syncId?: string;
  yAxisWidth?: number;
  ariaLabel?: string;
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
  totalLabel = i18n.t("shared.charts.stackedUsageChart.total"),
  scale = "linear",
  className,
  tooltipLabel = xLabel,
  tooltipFormat = format,
  showTotals = false,
  syncId,
  yAxisWidth,
  ariaLabel,
}: StackedUsageChartProps) {
  const fill = (index: number) => colors?.[index] ?? stackedUsageColor(index);
  const nameOf = (index: number) => labels?.[index] ?? series[index];
  const config = Object.fromEntries(
    series.map((key, index) => [key, { label: nameOf(index), color: fill(index) }]),
  ) satisfies ChartConfig;
  const tooltipHeader = (_label: unknown, payload: readonly { payload?: Record<string, unknown> }[]) => {
    const xValue = String(payload?.[0]?.payload?.[xKey] ?? "");
    const total = totalFor?.(xValue);
    return total === undefined
      ? tooltipLabel(xValue)
      : `${tooltipLabel(xValue)} · ${totalLabel} ${tooltipFormat(total)}`;
  };
  const totalsVisible = showTotals && scale === "linear" && totalFor !== undefined;
  const renderTotal = ({ x, y, width, index }: BarTotalLabelProps) => {
    const row = index === undefined ? undefined : data[index];
    const total = row === undefined ? undefined : totalFor?.(String(row[xKey] ?? ""));
    if (total === undefined) return null;
    return (
      <text
        x={Number(x) + Number(width) / 2}
        y={Number(y) - 6}
        textAnchor="middle"
        className="fill-muted-foreground text-[11px] tabular-nums"
      >
        {format(total)}
      </text>
    );
  };

  return (
    <ChartContainer
      config={config}
      role="group"
      aria-label={ariaLabel}
      className={cn("aspect-auto h-[380px] w-full", className)}
    >
      <BarChart
        data={[...data]}
        margin={{ left: 8, right: 8, top: totalsVisible ? 20 : 5 }}
        barCategoryGap="15%"
        maxBarSize={64}
        syncId={syncId}
      >
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
          domain={scale === "log" ? LOG_SCALE_DOMAIN : LINEAR_SCALE_DOMAIN}
          allowDataOverflow
          width={yAxisWidth}
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
                  format={tooltipFormat}
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
          >
            {totalsVisible && index === series.length - 1 && <LabelList dataKey={key} content={renderTotal} />}
          </Bar>
        ))}
      </BarChart>
    </ChartContainer>
  );
}
