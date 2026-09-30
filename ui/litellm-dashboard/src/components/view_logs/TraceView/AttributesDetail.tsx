"use client";

import type { Span } from "./traceTypes";

interface AttributesDetailProps {
  traceId: string;
  span: Span;
  attributes: Record<string, string> | undefined;
  isLoading: boolean;
}

/** Raw OTEL attributes as a key / value grid, ids first. */
export function AttributesDetail({ traceId, span, attributes, isLoading }: AttributesDetailProps) {
  const entries: [string, string][] = [
    ["trace_id", traceId],
    ["span_id", span.span_id],
    ["parent_span_id", span.parent_span_id ?? "—"],
    ...Object.entries(attributes ?? {}).sort(([a], [b]) => a.localeCompare(b)),
  ];
  return (
    <div className="p-3">
      <div className="overflow-hidden rounded border border-border bg-card font-mono text-[11px]">
        {entries.map(([key, value]) => (
          <div key={key} className="grid grid-cols-[minmax(120px,42%)_1fr] border-b border-border last:border-0">
            <div className="border-r border-border bg-muted/40 px-2.5 py-2 break-all text-muted-foreground">{key}</div>
            <div className="px-2.5 py-2 break-all text-foreground">{value}</div>
          </div>
        ))}
      </div>
      {isLoading && <div className="mt-2 font-mono text-[11px] text-muted-foreground">Loading attributes…</div>}
    </div>
  );
}
