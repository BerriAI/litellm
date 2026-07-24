"use client";

import * as React from "react";
import { Bar, BarChart as RechartsBarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, type ChartConfig } from "@/components/ui/chart";
import { cn } from "@/lib/cva.config";
import { ValueTooltip, type ChartTooltipComponent } from "./chart_tooltip";
import { categoryFills, type ChartColor } from "./colors";

export type BarChartProps<TDatum extends Record<string, unknown>> = {
  data: readonly TDatum[];
  index: string;
  categories: readonly string[];
  colors?: readonly ChartColor[];
  valueFormatter?: (value: number) => string;
  stack?: boolean;
  layout?: "horizontal" | "vertical";
  yAxisWidth?: number;
  tickGap?: number;
  showLegend?: boolean;
  showXAxis?: boolean;
  showGridLines?: boolean;
  showTooltip?: boolean;
  customTooltip?: ChartTooltipComponent;
  onValueChange?: (item: TDatum & { categoryClicked: string }) => void;
  className?: string;
  style?: React.CSSProperties;
};

export function BarChart<TDatum extends Record<string, unknown>>({
  data,
  index,
  categories,
  colors,
  valueFormatter,
  stack = false,
  layout = "horizontal",
  yAxisWidth = 56,
  tickGap = 5,
  showLegend = true,
  showXAxis = true,
  showGridLines = true,
  showTooltip = true,
  customTooltip,
  onValueChange,
  className,
  style,
}: BarChartProps<TDatum>) {
  const fills = categoryFills(categories.length, colors);
  const config: ChartConfig = Object.fromEntries(categories.map((category) => [category, { label: category }]));
  const vertical = layout === "vertical";
  const TooltipContent = customTooltip ?? ValueTooltip;

  return (
    <ChartContainer config={config} className={cn("aspect-auto h-80 w-full", className)} style={style}>
      <RechartsBarChart data={[...data]} layout={layout}>
        {showGridLines && <CartesianGrid horizontal={!vertical} vertical={vertical} />}
        {vertical ? (
          <XAxis
            type="number"
            hide={!showXAxis}
            tickLine={false}
            axisLine={false}
            minTickGap={tickGap}
            tickFormatter={valueFormatter}
          />
        ) : (
          <XAxis
            dataKey={index}
            hide={!showXAxis}
            tickLine={false}
            axisLine={false}
            minTickGap={tickGap}
            interval="equidistantPreserveStart"
          />
        )}
        {vertical ? (
          <YAxis type="category" dataKey={index} width={yAxisWidth} tickLine={false} axisLine={false} interval={0} />
        ) : (
          <YAxis width={yAxisWidth} tickLine={false} axisLine={false} tickFormatter={valueFormatter} />
        )}
        {showTooltip && (
          <ChartTooltip
            content={({ active, payload, label }) => (
              <TooltipContent
                active={active}
                payload={payload}
                label={label}
                {...(customTooltip ? {} : { valueFormatter })}
              />
            )}
          />
        )}
        {showLegend && (
          <ChartLegend
            verticalAlign="top"
            content={<ChartLegendContent className="justify-end text-muted-foreground" />}
          />
        )}
        {categories.map((category, i) => (
          <Bar
            key={category}
            dataKey={category}
            fill={fills[i]}
            stackId={stack ? "stack" : undefined}
            isAnimationActive={false}
            onClick={
              onValueChange
                ? (item: { payload?: TDatum }) => {
                    if (item.payload) onValueChange({ ...item.payload, categoryClicked: category });
                  }
                : undefined
            }
          />
        ))}
      </RechartsBarChart>
    </ChartContainer>
  );
}
