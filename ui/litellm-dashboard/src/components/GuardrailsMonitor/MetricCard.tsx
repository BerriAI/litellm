import React, { type ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: string | number;
  valueColor?: string;
  icon?: ReactNode;
  subtitle?: string;
}

export function MetricCard({ label, value, valueColor = "text-foreground", icon, subtitle }: MetricCardProps) {
  return (
    <div className="h-full bg-card border border-border rounded-lg p-5 flex flex-col">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-muted-foreground">{label}</span>
        {icon && <span className="text-muted-foreground">{icon}</span>}
      </div>
      <div className={`text-3xl font-semibold ${valueColor} tracking-tight`}>{value}</div>
      {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
    </div>
  );
}
