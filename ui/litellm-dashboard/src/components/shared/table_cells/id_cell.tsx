"use client";

import { Copy } from "lucide-react";
import * as React from "react";

import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import { CellTooltip } from "./cell_tooltip";

export type IdCellVariant = "pill" | "plain";

interface IdCellProps {
  value: string | null | undefined;
  variant?: IdCellVariant;
  onClick?: (value: string) => void;
  copyable?: boolean;
  truncate?: boolean;
  fallback?: string;
  tooltip?: React.ReactNode;
  disabled?: boolean;
  dataTestId?: string;
  className?: string;
}

const VARIANT_CLASS: Record<IdCellVariant, { base: string; clickable: string }> = {
  pill: {
    base: "font-mono text-xs font-normal px-2 py-0.5 rounded-md text-left bg-blue-50 text-blue-500",
    clickable: "hover:bg-blue-100 cursor-pointer",
  },
  plain: {
    base: "font-mono text-xs text-left",
    clickable: "hover:text-blue-600 cursor-pointer",
  },
};

export function IdCell({
  value,
  variant = "pill",
  onClick,
  copyable = false,
  truncate = true,
  fallback = "-",
  tooltip,
  disabled = false,
  dataTestId,
  className,
}: IdCellProps) {
  if (!value) {
    return <span className="text-muted-foreground">{fallback}</span>;
  }

  const clickable = !!onClick && !disabled;
  const classes = cn(
    VARIANT_CLASS[variant].base,
    clickable && VARIANT_CLASS[variant].clickable,
    truncate && "block max-w-[15ch] truncate",
    disabled && "opacity-50",
    className,
  );

  const idElement = clickable ? (
    <button type="button" className={classes} data-testid={dataTestId} onClick={() => onClick(value)}>
      {value}
    </button>
  ) : (
    <span className={classes} data-testid={dataTestId}>
      {value}
    </span>
  );

  const withTooltip = <CellTooltip content={tooltip ?? value} trigger={idElement} />;

  if (!copyable) {
    return withTooltip;
  }

  return (
    <span className="inline-flex max-w-full items-center gap-1">
      {withTooltip}
      <button
        type="button"
        aria-label="Copy ID"
        className="shrink-0 cursor-pointer text-muted-foreground hover:text-foreground"
        onClick={(event) => {
          event.stopPropagation();
          void copyToClipboard(value);
        }}
      >
        <Copy className="size-3" />
      </button>
    </span>
  );
}
