"use client";

import * as React from "react";

import { cn } from "@/lib/cva.config";

interface IdentityCellProps {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  className?: string;
  titleClassName?: string;
}

export function IdentityCell({ title, subtitle, className, titleClassName }: IdentityCellProps) {
  const showSubtitle = subtitle != null && subtitle !== "";
  return (
    <div className={cn("flex min-w-0 flex-col gap-0.5", className)}>
      <span className={cn("truncate text-sm font-medium text-foreground", titleClassName)}>{title}</span>
      {showSubtitle && <span className="truncate font-mono text-xs text-muted-foreground">{subtitle}</span>}
    </div>
  );
}
