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
      className="inline-flex min-w-0 items-center gap-1 rounded bg-muted px-1.5 py-px font-mono text-[11px] text-muted-foreground hover:text-foreground"
    >
      <span className="truncate">{showValue ? value : "ID"}</span>
      <Copy className="size-2.5 shrink-0" />
    </button>
  );
}
