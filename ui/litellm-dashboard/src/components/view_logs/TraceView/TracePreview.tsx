"use client";

import { cn } from "@/lib/cva.config";

import previewTrace from "./previewTrace.json";
import { SpanIcon } from "./SpanIcon";
import type { SpanType } from "./traceTypes";
import { fmtMs } from "./traceUtils";

interface PreviewRow {
  id: string;
  name: string;
  type: SpanType;
  model: string | null;
  depth: number;
  start_offset_ms: number;
  duration_ms: number;
  error: boolean;
}

const preview = previewTrace as {
  name: string;
  input_preview: string;
  span_count: number;
  duration_ms: number;
  rows: PreviewRow[];
};

export function TracePreview() {
  const total = preview.duration_ms;
  return (
    <div
      aria-hidden
      data-testid="trace-preview"
      className="pointer-events-none overflow-hidden rounded-lg border border-border bg-background shadow-sm select-none"
    >
      <div className="flex items-center gap-3 border-b border-border px-4 py-2.5">
        <SpanIcon type="agent" size="md" />
        <span className="text-[13px] font-medium text-foreground">{preview.name}</span>
        <span className="truncate text-[12px] text-muted-foreground">{preview.input_preview}</span>
        <span className="ml-auto shrink-0 font-mono text-[11px] text-muted-foreground">
          {preview.span_count} steps · {fmtMs(total)}
        </span>
      </div>
      <div className="relative">
        {preview.rows.map((row) => (
          <div key={row.id} className="flex h-8 items-center gap-2 border-b border-border/60 px-4 last:border-b-0">
            <div className="flex min-w-0 flex-1 items-center gap-2" style={{ paddingLeft: row.depth * 16 }}>
              <SpanIcon type={row.type} model={row.model} error={row.error} size="sm" />
              <span className="truncate text-[12.5px] text-foreground">{row.name}</span>
            </div>
            <div className="relative h-1.5 w-[40%] shrink-0 rounded-full bg-muted">
              <div
                className={cn("absolute h-full rounded-full", row.error ? "bg-destructive" : "bg-trace-brand/70")}
                style={{
                  left: `${(row.start_offset_ms / total) * 100}%`,
                  width: `${Math.max((row.duration_ms / total) * 100, 0.6)}%`,
                }}
              />
            </div>
            <span className="w-14 shrink-0 text-right font-mono text-[11px] text-muted-foreground">
              {fmtMs(row.duration_ms)}
            </span>
          </div>
        ))}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-background to-transparent" />
      </div>
    </div>
  );
}
