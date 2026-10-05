"use client";

import { Copy } from "lucide-react";

import { copyToClipboard } from "@/utils/dataUtils";

/** Small chip that copies an id; shows "ID" alone, or the id itself when `showValue` is set. */
export function IdChip({ value, label, showValue = false }: { value: string; label: string; showValue?: boolean }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={value}
      onClick={() => void copyToClipboard(value, "ID copied")}
      className="inline-flex h-6 min-w-0 shrink-0 items-center gap-1 rounded px-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
    >
      <span className="truncate">{showValue ? value : "ID"}</span>
      <Copy className="size-3 shrink-0" />
    </button>
  );
}
