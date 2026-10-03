import { Circle } from "lucide-react";
import { cn } from "@/lib/cva.config";

const colors = {
  ok: "fill-emerald-500 text-emerald-500",
  warn: "fill-amber-500 text-amber-500",
  off: "fill-slate-400 text-slate-400",
};

export function StatusDot({ state, className }: { state: keyof typeof colors; className?: string }) {
  return <Circle data-state={state} className={cn("size-2", colors[state], className)} />;
}
