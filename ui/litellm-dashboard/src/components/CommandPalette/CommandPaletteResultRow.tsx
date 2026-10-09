"use client";

import type { ReactNode } from "react";
import { cn } from "@/lib/cva.config";

export interface CommandPaletteRowData {
  id: string;
  title: string;
  subtitle?: ReactNode;
  icon: ReactNode;
  typeLabel: "Key" | "Page" | "Action";
}

export function CommandPaletteResultRow({
  item,
  optionId,
  active,
  onActivate,
  onHover,
}: {
  item: CommandPaletteRowData;
  optionId: string;
  active: boolean;
  onActivate: () => void;
  onHover: () => void;
}) {
  return (
    <button
      id={optionId}
      type="button"
      role="option"
      aria-selected={active}
      tabIndex={-1}
      className={cn(
        "flex h-11 w-full items-center gap-3 rounded-lg px-2 text-left transition-colors",
        active ? "bg-accent text-accent-foreground" : "hover:bg-muted/70",
      )}
      onMouseDown={(event) => event.preventDefault()}
      onClick={onActivate}
      onMouseMove={onHover}
    >
      <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-muted">{item.icon}</span>
      <span className="flex min-w-0 flex-1 items-baseline gap-2">
        <span className="truncate text-sm font-medium">{item.title}</span>
        {item.subtitle && <span className="truncate text-xs text-muted-foreground">{item.subtitle}</span>}
      </span>
      <span className="shrink-0 text-xs text-muted-foreground">{item.typeLabel}</span>
    </button>
  );
}
