"use client";

import * as React from "react";
import { Area, AreaChart as RechartsAreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, type ChartConfig } from "@/components/ui/chart";
import { cn } from "@/lib/cva.config";
import { ValueTooltip, type ChartTooltipComponent } from "./chart_tooltip";
import { categoryFills, type ChartColor } from "./colors";

export type AreaChartProps<TDatum extends Record<string, unknown>> = {
  data: readonly TDatum[];
  index: string;
  categories: readonly string[];
  colors?: readonly ChartColor[];
  valueFormatter?: (value: number) => string;
  yAxisWidth?: number;
  showLegend?: boolean;
  showGridLines?: boolean;
  showTooltip?: boolean;
  customTooltip?: ChartTooltipComponent;
  className?: string;
  style?: React.CSSProperties;
};

export function AreaChart<TDatum extends Record<string, unknown>>({
  data,
  index,
  categories,
  colors,
  valueFormatter,
  yAxisWidth = 56,
  showLegend = true,
  showGridLines = true,
  showTooltip = true,
  customTooltip,
  className,
  style,
}: AreaChartProps<TDatum>) {
  const gradientId = React.useId().replace(/:/g, "");
  const fills = categoryFills(categories.length, colors);
  const config: ChartConfig = Object.fromEntries(categories.map((category) => [category, { label: category }]));
  const TooltipContent = customTooltip ?? ValueTooltip;

  return (
    <ChartContainer config={config} className={cn("aspect-auto h-80 w-full", className)} style={style}>
      <RechartsAreaChart data={[...data]}>
        <defs>
          {categories.map((category, i) => (
            <linearGradient key={category} id={`fill-${gradientId}-${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={fills[i]} stopOpacity={0.4} />
              <stop offset="95%" stopColor={fills[i]} stopOpacity={0} />
            </linearGradient>
          ))}
        </defs>
        {showGridLines && <CartesianGrid vertical={false} />}
        <XAxis dataKey={index} tickLine={false} axisLine={false} minTickGap={5} interval="equidistantPreserveStart" />
        <YAxis width={yAxisWidth} tickLine={false} axisLine={false} tickFormatter={valueFormatter} />
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
          <Area
            key={category}
            type="linear"
            dataKey={category}
            stroke={fills[i]}
            strokeWidth={2}
            fill={`url(#fill-${gradientId}-${i})`}
            fillOpacity={1}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </RechartsAreaChart>
    </ChartContainer>
  );
}
