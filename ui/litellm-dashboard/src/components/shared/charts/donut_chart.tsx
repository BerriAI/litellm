"use client";

import * as React from "react";
import { Cell, Pie, PieChart } from "recharts";
import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart";
import { cn } from "@/lib/cva.config";
import { ValueTooltip } from "./chart_tooltip";
import { categoryFills, type ChartColor } from "./colors";

export type DonutChartProps<TDatum extends Record<string, unknown>> = {
  data: readonly TDatum[];
  index: string;
  category: string;
  colors?: readonly ChartColor[];
  variant?: "donut" | "pie";
  valueFormatter?: (value: number) => string;
  showTooltip?: boolean;
  showLabel?: boolean;
  label?: string;
  startAngle?: number;
  endAngle?: number;
  className?: string;
  style?: React.CSSProperties;
};

function formattedCategoryTotal<TDatum extends Record<string, unknown>>(
  data: readonly TDatum[],
  category: string,
  valueFormatter?: (value: number) => string,
): string {
  const total = data.reduce((sum, datum) => {
    const value = datum[category];
    return sum + (typeof value === "number" ? value : 0);
  }, 0);
  return valueFormatter ? valueFormatter(total) : String(total);
}

export function DonutChart<TDatum extends Record<string, unknown>>({
  data,
  index,
  category,
  colors,
  variant = "donut",
  valueFormatter,
  showTooltip = true,
  showLabel = false,
  label,
  startAngle = 0,
  endAngle = 360,
  className,
  style,
}: DonutChartProps<TDatum>) {
  const fills = categoryFills(data.length, colors);
  const config: ChartConfig = Object.fromEntries(
    data.map((datum, i) => {
      const name = String(datum[index] ?? i);
      return [name, { label: name }];
    }),
  );
  const showCenterLabel = showLabel && variant === "donut" && data.length > 0;

  return (
    <ChartContainer config={config} className={cn("aspect-auto h-40 w-full", className)} style={style}>
      <PieChart>
        {showTooltip && (
          <ChartTooltip
            content={({ active, payload, label: tooltipLabel }) => (
              <ValueTooltip active={active} payload={payload} label={tooltipLabel} valueFormatter={valueFormatter} />
            )}
          />
        )}
        {showCenterLabel && (
          <text className="fill-foreground text-base" x="50%" y="50%" textAnchor="middle" dominantBaseline="middle">
            {label ?? formattedCategoryTotal(data, category, valueFormatter)}
          </text>
        )}
        <Pie
          data={[...data]}
          dataKey={category}
          nameKey={index}
          innerRadius={variant === "pie" ? "0%" : "75%"}
          outerRadius="100%"
          startAngle={startAngle}
          endAngle={endAngle}
          strokeWidth={1}
          isAnimationActive={false}
        >
          {data.map((datum, i) => (
            <Cell key={String(datum[index] ?? i)} fill={fills[i]} />
          ))}
        </Pie>
      </PieChart>
    </ChartContainer>
  );
}
