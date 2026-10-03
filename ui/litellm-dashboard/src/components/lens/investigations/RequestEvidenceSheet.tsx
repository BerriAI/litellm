"use client";

import type { components } from "@/lib/http/schema";

import { Button } from "@/components/ui/button";

import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { evidenceTarget } from "../model/findings";

export function RequestEvidenceSheet({
  target,
  setEvidence,
  requestEvidence,
  requestOffset,
  setRequestOffset,
}: {
  target: ReturnType<typeof evidenceTarget>;
  setEvidence: (value: { id: string; span: string } | null) => void;
  requestEvidence: import("@tanstack/react-query").UseQueryResult<components["schemas"]["ExecutionContent"], Error>;
  requestOffset: number;
  setRequestOffset: (offset: number) => void;
}) {
  return (
    <Sheet
      open={target?.source === "requests"}
      onOpenChange={(open) => {
        if (!open) setEvidence(null);
      }}
    >
      <SheetContent className="overflow-y-auto data-[side=right]:sm:max-w-2xl">
        <SheetHeader>
          <SheetTitle>Request evidence</SheetTitle>
          <SheetDescription>Original logged input and output</SheetDescription>
        </SheetHeader>
        <div className="p-4 space-y-3">
          {requestEvidence.isLoading && <p role="status">Loading request…</p>}
          {requestEvidence.error && <p role="alert">{requestEvidence.error.message}</p>}
          {requestEvidence.data?.parts.map((p) => (
            <pre className="whitespace-pre-wrap break-words text-xs" key={p.span_id}>
              {p.content}
            </pre>
          ))}
          {requestEvidence.data?.parts.length === 0 && <p>Request was not found or is past retention</p>}
          <div className="flex flex-wrap gap-2">
            {requestOffset > 0 && (
              <Button variant="outline" onClick={() => setRequestOffset(Math.max(0, requestOffset - 8000))}>
                Previous section
              </Button>
            )}
            {requestEvidence.data?.parts.some((p) => p.truncated) && (
              <Button
                variant="outline"
                onClick={() => setRequestOffset(requestOffset === 0 ? 1 : requestOffset + 8000)}
              >
                Next section
              </Button>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
