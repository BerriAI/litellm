"use client";

import { ChevronRight } from "lucide-react";

import { cn } from "@/lib/cva.config";

/** Snaps between 0 and auto height; children stay mounted but inert while closed. */
export function Collapse({
  open,
  children,
  className,
}: {
  open: boolean;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("grid", open ? "grid-rows-[1fr]" : "grid-rows-[0fr]")} aria-hidden={!open} inert={!open}>
      <div className={cn("min-h-0 overflow-hidden", className)}>{children}</div>
    </div>
  );
}

/** Right-pointing chevron that rotates to point down when open. */
export function FoldChevron({ open, className }: { open: boolean; className?: string }) {
  return (
    <ChevronRight
      className={cn(
        "transition-transform duration-150 ease-[cubic-bezier(0.4,0,0.2,1)] motion-reduce:transition-none",
        open && "rotate-90",
        className,
      )}
    />
  );
}
