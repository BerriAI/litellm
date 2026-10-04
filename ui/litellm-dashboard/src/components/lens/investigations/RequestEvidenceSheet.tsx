"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";

import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { useLensApi } from "../LensServices";
import { lensQueries } from "../api/queries";
import type { EvidenceTarget } from "../model/findings";

const SECTION = 8000;

export interface RequestEvidenceSheetProps {
  readonly lensId: string | undefined;
  readonly target: EvidenceTarget | null;
  readonly onClose: () => void;
}

export function RequestEvidenceSheet({ lensId, target, onClose }: RequestEvidenceSheetProps) {
  const open = target?.source === "requests";
  return (
    <Sheet
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
    >
      <SheetContent className="overflow-y-auto data-[side=right]:sm:max-w-2xl">
        <SheetHeader>
          <SheetTitle>Request evidence</SheetTitle>
          <SheetDescription>Original logged input and output</SheetDescription>
        </SheetHeader>
        {open && lensId && <RequestContent key={target.id} lensId={lensId} evidenceId={target.id} />}
      </SheetContent>
    </Sheet>
  );
}

function RequestContent({ lensId, evidenceId }: { lensId: string; evidenceId: string }) {
  const api = useLensApi();
  const [offset, setOffset] = useState(0);
  const request = { lensId, evidenceId, requestOffset: offset, source: "requests" };
  const evidence = useQuery(lensQueries.evidence(api, request));
  return (
    <div className="p-4 space-y-3">
      {evidence.isLoading && <p role="status">Loading request…</p>}
      {evidence.error && <p role="alert">{evidence.error.message}</p>}
      {evidence.data?.parts.map((p) => (
        <pre className="whitespace-pre-wrap break-words text-xs" key={p.span_id}>
          {p.content}
        </pre>
      ))}
      {evidence.data?.parts.length === 0 && <p>Request was not found or is past retention</p>}
      <div className="flex flex-wrap gap-2">
        {offset > 0 && (
          <Button variant="outline" onClick={() => setOffset(Math.max(0, offset - SECTION))}>
            Previous section
          </Button>
        )}
        {evidence.data?.parts.some((p) => p.truncated) && (
          <Button variant="outline" onClick={() => setOffset(offset === 0 ? 1 : offset + SECTION)}>
            Next section
          </Button>
        )}
      </div>
    </div>
  );
}
