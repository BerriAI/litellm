"use client";

import type { components } from "@/lib/http/schema";

import { Button } from "@/components/ui/button";

import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { evidenceTarget } from "../model/findings";

type RequestEvidence = components["schemas"]["ExecutionContent"];

export function RequestEvidenceSheet({
  target,
  setEvidence,
  requestEvidenceData,
  requestEvidenceError,
  requestEvidenceLoading,
  requestOffset,
  setRequestOffset,
}: {
  target: ReturnType<typeof evidenceTarget>;
  setEvidence: (value: { id: string; span: string } | null) => void;
  requestEvidenceData: RequestEvidence | undefined;
  requestEvidenceError: Error | null | undefined;
  requestEvidenceLoading: boolean;
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
          {requestEvidenceLoading && <p role="status">Loading request…</p>}
          {requestEvidenceError && <p role="alert">{requestEvidenceError.message}</p>}
          {requestEvidenceData?.parts.map((p) => (
            <pre className="whitespace-pre-wrap break-words text-xs" key={p.span_id}>
              {p.content}
            </pre>
          ))}
          {requestEvidenceData?.parts.length === 0 && <p>Request was not found or is past retention</p>}
          <div className="flex flex-wrap gap-2">
            {requestOffset > 0 && (
              <Button variant="outline" onClick={() => setRequestOffset(Math.max(0, requestOffset - 8000))}>
                Previous section
              </Button>
            )}
            {requestEvidenceData?.parts.some((p) => p.truncated) && (
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
