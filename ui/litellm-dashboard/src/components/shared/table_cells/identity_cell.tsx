"use client";

import { ChevronRight } from "lucide-react";
import * as React from "react";

import { cn } from "@/lib/cva.config";

interface IdentityCellProps {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  badge?: React.ReactNode;
  onClick?: () => void;
  className?: string;
  titleClassName?: string;
}

export function IdentityCell({ title, subtitle, badge, onClick, className, titleClassName }: IdentityCellProps) {
  const hasSubtitleRow = (subtitle != null && subtitle !== "") || badge != null;

  const body = (
    <div className="flex min-w-0 flex-col gap-0.5">
      <span className={cn("truncate text-sm font-medium text-foreground", titleClassName)}>{title}</span>
      {hasSubtitleRow && (
        <span className="flex min-w-0 items-center gap-2">
          {subtitle != null && subtitle !== "" && (
            <span className="truncate font-mono text-xs text-muted-foreground">{subtitle}</span>
          )}
          {badge}
        </span>
      )}
    </div>
  );

  if (onClick != null) {
    return (
      <button
        type="button"
        onClick={onClick}
        className={cn("group flex w-full items-center gap-2 rounded-md py-1 text-left", className)}
      >
        {body}
        <ChevronRight className="ml-auto size-4 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
      </button>
    );
  }

  return <div className={cn("min-w-0", className)}>{body}</div>;
}
