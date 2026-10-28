"use client";

import React, { type ReactNode } from "react";
import { Radio as RadioPrimitive } from "@base-ui/react/radio";
import { RadioGroup as RadioGroupPrimitive } from "@base-ui/react/radio-group";
import { Area, AreaChart } from "recharts";
import type { LucideIcon } from "lucide-react";
import { ChartContainer } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cva.config";
import type { SeriesDay } from "./overviewData";

const EMPTY_CONFIG = {};

export const PANEL_INSET_X = "px-5";

export function Panel({
  icon: Icon,
  title,
  subtitle,
  action,
  className,
  bodyClassName,
  children,
  testId,
}: {
  icon?: LucideIcon;
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
  testId?: string;
}) {
  return (
    <section data-testid={testId} className={cn("flex min-w-0 flex-col rounded-xl border bg-card", className)}>
      <header
        className={cn(
          "flex min-h-8 flex-wrap justify-between gap-x-3 gap-y-2 pt-4",
          PANEL_INSET_X,
          subtitle ? "items-start" : "items-center",
        )}
      >
        <div className="min-w-0">
          <h3 className="flex min-w-0 items-center gap-2 text-sm leading-5 font-medium text-foreground">
            {Icon && <Icon aria-hidden="true" className="size-4 shrink-0 text-muted-foreground" strokeWidth={1.75} />}
            <span className="truncate">{title}</span>
          </h3>
          {subtitle && <p className="mt-0.5 truncate text-xs text-muted-foreground">{subtitle}</p>}
        </div>
        {action && <div className="flex flex-wrap items-center gap-2">{action}</div>}
      </header>
      <div className={cn("flex min-h-0 flex-1 flex-col pt-3 pb-4", PANEL_INSET_X, bodyClassName)}>{children}</div>
    </section>
  );
}

export function Sparkline({
  data,
  dataKey,
  color,
  className,
}: {
  data: readonly SeriesDay[];
  dataKey: string;
  color: string;
  className?: string;
}) {
  const gradientId = React.useId().replace(/:/g, "");
  if (data.length < 2) return <div className={cn("h-10 w-full", className)} />;
  return (
    // Decorative: no focus ring, hover cursor or tooltip, since the number above it is the readout.
    <ChartContainer
      config={EMPTY_CONFIG}
      aria-hidden="true"
      className={cn("pointer-events-none aspect-auto h-10 w-full", className)}
    >
      <AreaChart data={[...data]} margin={{ top: 2, right: 0, bottom: 0, left: 0 }} accessibilityLayer={false}>
        <defs>
          <linearGradient id={`spark-${gradientId}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.18} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          dataKey={dataKey}
          type="monotone"
          stroke={color}
          strokeWidth={1.5}
          fill={`url(#spark-${gradientId})`}
          isAnimationActive={false}
        />
      </AreaChart>
    </ChartContainer>
  );
}

export function ChartSkeleton({ className }: { className?: string }) {
  return <Skeleton className={cn("h-48 w-full rounded-lg", className)} />;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <RadioGroupPrimitive
      aria-label={label}
      value={value}
      onValueChange={(next) => onChange(next as T)}
      className="inline-flex items-center rounded-lg bg-muted p-0.5"
    >
      {options.map((option) => (
        <RadioPrimitive.Root
          key={option.value}
          value={option.value}
          className="rounded-md px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 data-checked:bg-background data-checked:text-foreground data-checked:shadow-xs"
        >
          {option.label}
        </RadioPrimitive.Root>
      ))}
    </RadioGroupPrimitive>
  );
}

export function Stat({
  label,
  value,
  pending,
  hint,
  tone,
  size = "md",
  exact,
}: {
  label: ReactNode;
  value: ReactNode;
  pending?: boolean;
  hint?: ReactNode;
  tone?: "success" | "destructive";
  size?: "md" | "lg";
  /** Full-precision value shown on hover when `value` is abbreviated (41.7M -> 41,719,305). */
  exact?: string;
}) {
  return (
    <div className="min-w-0">
      <div className="flex h-4 items-center gap-1.5 text-xs leading-4 text-muted-foreground">{label}</div>
      {pending ? (
        <Skeleton className={cn("mt-1", size === "lg" ? "h-9 w-32" : "h-7 w-20")} />
      ) : (
        <div
          title={exact}
          className={cn(
            "mt-1 truncate font-semibold tracking-tight tabular-nums",
            size === "lg" ? "text-3xl leading-9" : "text-xl leading-7",
            tone === "success" && "text-success",
            tone === "destructive" && "text-destructive",
          )}
        >
          {value}
        </div>
      )}
      {hint && <div className="mt-0.5 truncate text-xs leading-4 text-muted-foreground">{hint}</div>}
    </div>
  );
}
