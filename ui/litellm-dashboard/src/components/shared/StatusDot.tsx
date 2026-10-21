import type { ComponentProps } from "react";
import { Circle } from "lucide-react";
import { cn } from "@/lib/cva.config";

const colors = {
  ok: "fill-emerald-500 text-emerald-500",
  warn: "fill-amber-500 text-amber-500",
  error: "fill-red-500 text-red-500",
  off: "fill-slate-400 text-slate-400",
};

export type StatusDotProps = Omit<ComponentProps<typeof Circle>, "className"> & {
  state: keyof typeof colors;
  className?: string;
};

export function StatusDot({ state, className, ...props }: StatusDotProps) {
  return (
    <Circle {...props} data-slot="status-dot" data-state={state} className={cn("size-2", colors[state], className)} />
  );
}
