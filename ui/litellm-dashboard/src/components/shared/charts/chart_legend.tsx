"use client";

import * as React from "react";
import { formatCategoryName } from "./chart_tooltip";
import { chartColorValue, type ChartColor } from "./colors";

export const CustomLegend = ({
  categories,
  colors,
}: {
  categories: readonly string[];
  colors: readonly ChartColor[];
}) => (
  <div className="flex items-center justify-end space-x-4">
    {categories.map((category, idx) => (
      <div key={category} className="flex items-center space-x-2">
        <span
          className="h-2 w-2 shrink-0 rounded-full ring-4 ring-white"
          style={{ backgroundColor: chartColorValue(colors[idx % colors.length]) }}
        />
        <p className="text-sm text-muted-foreground">{formatCategoryName(category)}</p>
      </div>
    ))}
  </div>
);
