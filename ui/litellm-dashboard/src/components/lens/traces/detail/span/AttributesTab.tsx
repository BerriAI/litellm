"use client";

import { fieldEntries } from "../content/payload";
import type { Span } from "../../types";
import { FieldTree } from "../content/FieldTree";
import { DetailGroup } from "./DetailGroup";

interface AttributesTabProps {
  traceId: string;
  span: Span;
  attributes: Record<string, string> | undefined;
  isLoading: boolean;
}

/** Ids, then the raw OTEL attributes, as a key / value tree. */
export function AttributesTab({ traceId, span, attributes, isLoading }: AttributesTabProps) {
  const ids = fieldEntries([
    ["trace_id", traceId],
    ["span_id", span.span_id],
    ["parent_span_id", span.parent_span_id ?? "—"],
  ]);
  const attributeEntries = fieldEntries(Object.entries(attributes ?? {}).sort(([a], [b]) => a.localeCompare(b)));
  return (
    <div className="flex flex-col gap-5 px-4 pt-3 pb-5">
      <DetailGroup title="Identifiers">
        <FieldTree entries={ids} mono />
      </DetailGroup>
      {attributeEntries.length > 0 && (
        <DetailGroup title="Attributes">
          <FieldTree entries={attributeEntries} />
        </DetailGroup>
      )}
      {isLoading && <div className="text-sm text-muted-foreground">Loading attributes…</div>}
    </div>
  );
}
