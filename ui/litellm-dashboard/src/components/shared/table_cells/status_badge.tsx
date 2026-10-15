"use client";

import * as React from "react";

import { useEntityLinkClick } from "@/components/shared/EntityLink";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";

import { CellTooltip } from "./cell_tooltip";

export type StatusTone = "success" | "error" | "warning" | "neutral" | "info";

const TONE_CLASS: Record<StatusTone, string> = {
  success: "border-success/20 bg-success/10 text-success",
  error: "border-destructive/20 bg-destructive/10 text-destructive",
  warning: "border-warning/20 bg-warning/10 text-warning",
  neutral: "border-border bg-muted text-muted-foreground",
  info: "border-info/20 bg-info/10 text-info",
};

interface StatusBadgeProps {
  tone: StatusTone;
  label: string;
  tooltip?: React.ReactNode;
  dataTestId?: string;
  className?: string;
  href?: string;
}

export function StatusBadge({ tone, label, tooltip, dataTestId, className, href }: StatusBadgeProps) {
  const badgeClassName = cn("whitespace-nowrap font-normal", TONE_CLASS[tone], className);
  const badge = href ? (
    <LinkedStatusBadge href={href} dataTestId={dataTestId} className={badgeClassName}>
      {label}
    </LinkedStatusBadge>
  ) : (
    <Badge variant="outline" data-testid={dataTestId} className={badgeClassName}>
      {label}
    </Badge>
  );

  if (!tooltip) {
    return badge;
  }
  return <CellTooltip content={tooltip} trigger={badge} />;
}

interface LinkedStatusBadgeProps {
  href: string;
  dataTestId?: string;
  className: string;
  children: string;
}

function LinkedStatusBadge({ href, dataTestId, className, children }: LinkedStatusBadgeProps) {
  const handleClick = useEntityLinkClick(href);

  return (
    <Badge
      variant="outline"
      data-testid={dataTestId}
      className={cn("cursor-pointer hover:underline", className)}
      render={<a href={href} onClick={handleClick} />}
    >
      {children}
    </Badge>
  );
}
