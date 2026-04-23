import React, { type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string | number;
  /**
   * Tailwind text-color class. Defaults to a foreground-tone token.
   * Callers can pass any tone here including non-semantic ones (the
   * label/value usage is decorative, not a theme state).
   */
  valueColor?: string;
  icon?: ReactNode;
  subtitle?: string;
}

export function MetricCard({
  label,
  value,
  valueColor = "text-foreground",
  icon,
  subtitle,
}: MetricCardProps) {
  return (
    <div className="h-full bg-background border border-border rounded-lg p-5 flex flex-col">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-muted-foreground">
          {label}
        </span>
        {icon && <span className="text-muted-foreground">{icon}</span>}
      </div>
      <div className={cn("text-3xl font-semibold tracking-tight", valueColor)}>
        {value}
      </div>
      {subtitle && (
        <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
      )}
    </div>
  );
}
