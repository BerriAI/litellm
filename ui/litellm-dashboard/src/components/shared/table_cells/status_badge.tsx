"use client";

import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";

import { CellTooltip } from "./cell_tooltip";

export type StatusTone = "success" | "error" | "warning" | "neutral" | "info";

const DOT_CLASS: Record<StatusTone, string> = {
  success: "bg-emerald-500",
  error: "bg-red-500",
  warning: "bg-amber-500",
  neutral: "bg-gray-400",
  info: "bg-blue-500",
};

interface StatusBadgeProps {
  tone: StatusTone;
  label: string;
  tooltip?: React.ReactNode;
  dataTestId?: string;
}

export function StatusBadge({ tone, label, tooltip, dataTestId }: StatusBadgeProps) {
  const badge = (
    <Badge
      variant="outline"
      data-testid={dataTestId}
      className="gap-1.5 whitespace-nowrap font-normal text-muted-foreground"
    >
      <span aria-hidden="true" className={cn("size-1.5 rounded-full", DOT_CLASS[tone])} />
      {label}
    </Badge>
  );

  if (!tooltip) {
    return badge;
  }
  return <CellTooltip content={tooltip} trigger={badge} />;
}
