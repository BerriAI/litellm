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
      className="inline-flex h-5 min-w-0 shrink-0 items-center gap-1 rounded-[3px] border border-transparent bg-trace-chip px-1 font-mono text-[13px] leading-none font-medium text-trace-key transition-colors duration-100 hover:text-trace-text-2 focus-visible:outline-2 focus-visible:outline-trace-brand motion-reduce:transition-none"
    >
      <span className="truncate">{showValue ? value : "ID"}</span>
      <Copy className="size-3 shrink-0" />
    </button>
  );
}
