import { cn } from "@/lib/cva.config";

interface DurationBarProps {
  startMs: number;
  durationMs: number;
  totalMs: number;
  error?: boolean;
  className?: string;
}

/** Waterfall bar: a hairline track with the span's slice of the run's timeline. */
export function DurationBar({ startMs, durationMs, totalMs, error = false, className }: DurationBarProps) {
  const left = totalMs > 0 ? Math.max(0, Math.min(98, (startMs / totalMs) * 100)) : 0;
  const width = totalMs > 0 ? Math.max(1.5, Math.min(100 - left, (durationMs / totalMs) * 100)) : 1.5;
  return (
    <div className={cn("relative h-3 w-full overflow-hidden", className)} aria-hidden="true">
      <div className="absolute top-[5px] left-0 h-px w-full bg-border" />
      <div
        className={cn(
          "absolute top-[3px] h-[5px] min-w-[2px] rounded-[1px]",
          error ? "bg-destructive/70" : "bg-muted-foreground/60",
        )}
        style={{ left: `${left}%`, width: `${width}%` }}
      />
    </div>
  );
}
