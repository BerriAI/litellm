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

export function DetailGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section aria-label={title} className="flex flex-col gap-2">
      <h3 className="py-1 text-sm font-medium text-trace-text">{title}</h3>
      <Card className="px-3 py-2.5">{children}</Card>
    </section>
  );
}

/** Ids, then the raw OTEL attributes, as dot-bulleted key / value rows. */
export function AttributesDetail({ traceId, span, attributes, isLoading }: AttributesDetailProps) {
  const ids: KeyValue[] = [
    ["trace_id", traceId],
    ["span_id", span.span_id],
    ["parent_span_id", span.parent_span_id ?? "—"],
  ];
  const attributeEntries: KeyValue[] = Object.entries(attributes ?? {}).sort(([a], [b]) => a.localeCompare(b));
  return (
    <div className="flex flex-col gap-4 px-7 pt-1 pb-4">
      <DetailGroup title="Identifiers">
        <KeyValueRows entries={ids} mono />
      </DetailGroup>
      {attributeEntries.length > 0 && (
        <DetailGroup title="Attributes">
          <KeyValueRows entries={attributeEntries} />
        </DetailGroup>
      )}
      {isLoading && <div className="text-sm text-trace-duration">Loading attributes…</div>}
    </div>
  );
}
