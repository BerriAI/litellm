"use client";

import * as React from "react";
import { useTranslation } from "react-i18next";
import { formatCategoryName } from "./chart_tooltip";
import { chartColorValue, type ChartColor } from "./colors";

export const CustomLegend = ({
  categories,
  categoryLabels,
  colors,
}: {
  categories: readonly string[];
  categoryLabels?: Readonly<Record<string, string>>;
  colors: readonly ChartColor[];
}) => {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center justify-end gap-x-4 gap-y-1">
      {categories.map((category, idx) => (
        <div key={category} className="flex items-center space-x-2">
          <span
            className="h-2 w-2 shrink-0 rounded-full ring-4 ring-white"
            style={{ backgroundColor: chartColorValue(colors[idx % colors.length]) }}
          />
          <p className="text-sm text-muted-foreground">
            {categoryLabels?.[category] ?? formatCategoryName(category, t)}
          </p>
        </div>
      ))}
    </div>
  );
};
