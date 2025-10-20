import React from "react";
import type { CustomTooltipProps } from "@tremor/react";
import { SpendMetrics } from "../usage/types";

interface ChartDataPoint {
  date: string;
  metrics: SpendMetrics;
}

const colorNameToHex: { [key: string]: string } = {
  blue: "#3b82f6",
  cyan: "#06b6d4",
  indigo: "#6366f1",
  green: "#22c55e",
  red: "#ef4444",
  purple: "#8b5cf6",
};

export const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    const formatCategoryName = (name: string): string => {
      return name
        .replace("metrics.", "")
        .replace(/_/g, " ")
        .split(" ")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
    };

    const getRawValue = (dataPoint: ChartDataPoint, key: string): number | undefined => {
      // key is like "metrics.total_tokens"
      const metricKey = key.substring(key.indexOf(".") + 1) as keyof SpendMetrics;
      if (dataPoint.metrics && metricKey in dataPoint.metrics) {
        return dataPoint.metrics[metricKey];
      }
      return undefined;
    };

    return (
      <div className="w-56 rounded-tremor-default border border-tremor-border bg-tremor-background p-2 text-tremor-default shadow-tremor-dropdown">
        <p className="text-tremor-content-strong">{label}</p>
        {payload.map((item) => {
          const dataKey = item.dataKey?.toString();
          if (!dataKey || !item.payload) return null;

          const rawValue = getRawValue(item.payload, dataKey);
          const isSpend = dataKey.includes("spend");
          const formattedValue =
            rawValue !== undefined
              ? isSpend
                ? `$${rawValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                : rawValue.toLocaleString()
              : "N/A";

          const colorName = item.color as keyof typeof colorNameToHex;
          const hexColor = colorNameToHex[colorName] || item.color;
          return (
            <div key={dataKey} className="flex items-center justify-between space-x-4">
              <div className="flex items-center space-x-2">
                <span
                  className={`h-2 w-2 shrink-0 rounded-full ring-2 ring-white drop-shadow-md`}
                  style={{ backgroundColor: hexColor }}
                />
                <p className="font-medium text-tremor-content dark:text-dark-tremor-content">
                  {formatCategoryName(dataKey)}
                </p>
              </div>
              <p className="font-medium text-tremor-content-emphasis dark:text-dark-tremor-content-emphasis">
                {formattedValue}
              </p>
            </div>
          );
        })}
      </div>
    );
  }
  return null;
};

export const CustomLegend = ({ categories, colors }: { categories: string[]; colors: string[] }) => {
  const formatCategoryName = (name: string): string => {
    return name
      .replace("metrics.", "")
      .replace(/_/g, " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  return (
    <div className="flex items-center justify-end space-x-4">
      {categories.map((category, idx) => {
        const colorName = colors[idx] as keyof typeof colorNameToHex;
        const hexColor = colorNameToHex[colorName] || colors[idx];
        return (
          <div key={category} className="flex items-center space-x-2">
            <span className={`h-2 w-2 shrink-0 rounded-full ring-4 ring-white`} style={{ backgroundColor: hexColor }} />
            <p className="text-sm text-tremor-content dark:text-dark-tremor-content">{formatCategoryName(category)}</p>
          </div>
        );
      })}
    </div>
  );
};
