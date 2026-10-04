import { cn } from "@/lib/cva.config";

export function PaneBar({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("flex h-10 shrink-0 items-center gap-2 border-b px-3", className)}>{children}</div>;
}
