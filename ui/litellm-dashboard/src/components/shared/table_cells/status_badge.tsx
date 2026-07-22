"use client";

import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";

import { CellTooltip } from "./cell_tooltip";

export type StatusTone = "success" | "error" | "warning" | "neutral" | "info";

const TONE_CLASS: Record<StatusTone, string> = {
  success: "border-green-200 bg-green-50 text-green-600",
  error: "border-red-200 bg-red-50 text-red-600",
  warning: "border-amber-200 bg-amber-50 text-amber-600",
  neutral: "border-gray-200 bg-gray-50 text-gray-600",
  info: "border-blue-200 bg-blue-50 text-blue-600",
};

interface StatusBadgeProps {
  tone: StatusTone;
  label: string;
  tooltip?: React.ReactNode;
  dataTestId?: string;
}

export function StatusBadge({ tone, label, tooltip, dataTestId }: StatusBadgeProps) {
  const badge = (
    <Badge variant="outline" data-testid={dataTestId} className={cn("whitespace-nowrap font-normal", TONE_CLASS[tone])}>
      {label}
    </Badge>
  );

  if (!tooltip) {
    return badge;
  }
  return <CellTooltip content={tooltip} trigger={badge} />;
}
