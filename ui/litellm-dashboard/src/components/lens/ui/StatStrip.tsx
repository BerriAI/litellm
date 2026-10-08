"use client";

import { useId, type ReactNode } from "react";
import { Area, AreaChart } from "recharts";

import { ChartContainer } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cva.config";

const BRAND_BLUE = "#2b3fd6";
const EMPTY_CONFIG = {};

export function StatStrip({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "grid shrink-0 grid-cols-2 overflow-hidden rounded-xl border bg-card max-lg:[&>*:nth-child(n+3)]:border-t lg:grid-cols-4 lg:divide-x max-lg:[&>*:nth-child(even)]:border-l",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function StatCell({
  label,
  value,
  hint,
  title,
  pending,
  spark,
}: {
  label: string;
  value: ReactNode;
  hint: ReactNode;
  title?: string;
  pending?: boolean;
  spark?: readonly number[];
}) {
  return (
    <div className="flex min-w-0 flex-col px-4 pt-3.5 pb-3" role="group" aria-label={label}>
      <div className="text-xs text-muted-foreground">{label}</div>
      {pending ? (
        <Skeleton className="mt-1 h-7 w-20" />
      ) : (
        <div title={title} className="mt-1 truncate text-xl leading-7 font-semibold tracking-tight tabular-nums">
          {value}
        </div>
      )}
      <div className="mt-0.5 truncate text-xs text-muted-foreground">{pending ? " " : hint}</div>
      {spark && <Sparkline values={pending ? [] : spark} className="mt-2" />}
    </div>
  );
}

export function Sparkline({ values, className }: { values: readonly number[]; className?: string }) {
  const gradientId = `lens-spark-${useId().replace(/:/g, "")}`;
  if (values.length < 2 || values.every((value) => value === 0))
    return <div aria-hidden="true" className={cn("h-14 w-full", className)} />;
  return (
    <ChartContainer
      config={EMPTY_CONFIG}
      aria-hidden="true"
      className={cn("pointer-events-none aspect-auto h-14 w-full", className)}
    >
      <AreaChart
        data={values.map((value, index) => ({ index, value }))}
        margin={{ top: 2, right: 0, bottom: 0, left: 0 }}
        accessibilityLayer={false}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={BRAND_BLUE} stopOpacity={0.18} />
            <stop offset="100%" stopColor={BRAND_BLUE} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          dataKey="value"
          type="monotone"
          stroke={BRAND_BLUE}
          strokeWidth={1.5}
          fill={`url(#${gradientId})`}
          isAnimationActive={false}
        />
      </AreaChart>
    </ChartContainer>
  );
}
