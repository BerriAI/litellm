import React, { type ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: string | number;
  valueColor?: string;
  icon?: ReactNode;
  subtitle?: string;
}

export function MetricCard({
  label,
  value,
  valueColor = "text-gray-900",
  icon,
  subtitle,
}: MetricCardProps) {
  return (
    <div className="h-full bg-white border border-gray-200 rounded-lg p-5 flex flex-col">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-gray-600">{label}</span>
        {icon && <span className="text-gray-400">{icon}</span>}
      </div>
      <div className={`text-3xl font-semibold ${valueColor} tracking-tight`}>
        {value}
      </div>
      {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
    </div>
  );
}
