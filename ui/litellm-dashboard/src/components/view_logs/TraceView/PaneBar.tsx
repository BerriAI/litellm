import { cn } from "@/lib/cva.config";

/** Fixed-height pane row; every pane in the run view stacks these so borders line up across the split. */
export function PaneBar({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("flex h-10 shrink-0 items-center gap-2 border-b px-3", className)}>{children}</div>;
}
