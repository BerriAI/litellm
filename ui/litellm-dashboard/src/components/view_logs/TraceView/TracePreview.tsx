"use client";

import { useMemo } from "react";

import previewTrace from "./__fixtures__/deep_agent_trace.json";
import { SpanIcon } from "./SpanIcon";
import type { SpanRowData } from "./traceTree";
import type { Trace } from "./traceTypes";
import { buildTreeRows, fmtMs } from "./traceUtils";

const PREVIEW_ROWS = 9;
const PREVIEW_STATE = {
  hideFramework: true,
  collapsedSpanIds: new Set<string>(),
  expandedGroupIds: new Set<string>(),
  groupRevealCounts: {},
};

const trace = previewTrace as unknown as Trace;

/** A static, crisp rendering of a real deep-agent run, so the empty state shows what Agent Traces looks like. */
export function TracePreview() {
  const rows = useMemo(
    () =>
      buildTreeRows(trace.spans, PREVIEW_STATE)
        .filter((row): row is SpanRowData => row.kind === "span")
        .slice(0, PREVIEW_ROWS),
    [],
  );
  const total = trace.summary.duration_ms;
  return (
    <div
      aria-hidden
      data-testid="trace-preview"
      className="pointer-events-none overflow-hidden rounded-lg border border-border bg-background shadow-sm select-none"
    >
      <div className="flex items-center gap-3 border-b border-border px-4 py-2.5">
        <SpanIcon type="agent" size="md" />
        <span className="text-[13px] font-medium text-foreground">{trace.summary.name}</span>
        <span className="truncate text-[12px] text-muted-foreground">{trace.summary.input_preview}</span>
        <span className="ml-auto shrink-0 font-mono text-[11px] text-muted-foreground">
          {trace.summary.span_count} steps · {fmtMs(total)}
        </span>
      </div>
      <div className="relative">
        {rows.map((row) => (
          <div key={row.id} className="flex h-8 items-center gap-2 border-b border-border/60 px-4 last:border-b-0">
            <div className="flex min-w-0 flex-1 items-center gap-2" style={{ paddingLeft: row.depth * 16 }}>
              <SpanIcon type={row.span.type} model={row.span.model} error={row.span.status === "error"} size="sm" />
              <span className="truncate text-[12.5px] text-foreground">{row.span.name}</span>
            </div>
            <div className="relative h-1.5 w-[40%] shrink-0 rounded-full bg-muted">
              <div
                className={
                  row.span.status === "error"
                    ? "absolute h-full rounded-full bg-destructive"
                    : "absolute h-full rounded-full bg-trace-brand/70"
                }
                style={{
                  left: `${(row.span.start_offset_ms / total) * 100}%`,
                  width: `${Math.max((row.span.duration_ms / total) * 100, 0.6)}%`,
                }}
              />
            </div>
            <span className="w-14 shrink-0 text-right font-mono text-[11px] text-muted-foreground">
              {fmtMs(row.span.duration_ms)}
            </span>
          </div>
        ))}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-background to-transparent" />
      </div>
    </div>
  );
}
