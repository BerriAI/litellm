"use client";

import { type KeyValue, KeyValueRows } from "./KeyValueRows";
import { Card } from "./MessageCard";
import type { Span } from "./traceTypes";

interface AttributesDetailProps {
  traceId: string;
  span: Span;
  attributes: Record<string, string> | undefined;
  isLoading: boolean;
}

/** Raw OTEL attributes as key / value rows, ids first. */
export function AttributesDetail({ traceId, span, attributes, isLoading }: AttributesDetailProps) {
  const entries: KeyValue[] = [
    ["trace_id", traceId],
    ["span_id", span.span_id],
    ["parent_span_id", span.parent_span_id ?? "—"],
    ...Object.entries(attributes ?? {}).sort(([a], [b]) => a.localeCompare(b)),
  ];
  return (
    <div className="px-5 py-3">
      <Card>
        <KeyValueRows entries={entries} mono />
      </Card>
      {isLoading && <div className="mt-2 text-[12px] text-muted-foreground">Loading attributes…</div>}
    </div>
  );
}
