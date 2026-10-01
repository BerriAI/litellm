"use client";

import { ChevronRight } from "lucide-react";

import { cn } from "@/lib/cva.config";

/** Animates height between 0 and auto via grid rows; children stay mounted but inert while closed. */
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
    <div
      className={cn(
        "grid transition-[grid-template-rows,opacity] duration-150 ease-[cubic-bezier(0,0,0.2,1)] motion-reduce:transition-none",
        open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
      )}
      aria-hidden={!open}
      inert={!open}
    >
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
