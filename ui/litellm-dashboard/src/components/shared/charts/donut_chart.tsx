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
  className?: string;
  style?: React.CSSProperties;
};

export function DonutChart<TDatum extends Record<string, unknown>>({
  data,
  index,
  category,
  colors,
  variant = "donut",
  valueFormatter,
  showTooltip = true,
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

  return (
    <ChartContainer config={config} className={cn("aspect-auto h-40 w-full", className)} style={style}>
      <PieChart>
        {showTooltip && (
          <ChartTooltip
            content={({ active, payload, label }) => (
              <ValueTooltip active={active} payload={payload} label={label} valueFormatter={valueFormatter} />
            )}
          />
        )}
        <Pie
          data={[...data]}
          dataKey={category}
          nameKey={index}
          innerRadius={variant === "pie" ? "0%" : "75%"}
          outerRadius="100%"
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
