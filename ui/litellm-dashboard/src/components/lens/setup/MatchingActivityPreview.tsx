"use client";

import { ChevronRight, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";

import { type Sample } from "../model/types";
import { runTime } from "../model/format";

export function RunList({ executions }: { executions: Sample["executions"] }) {
  return (
    <div className="divide-y">
      {executions.map((run) => (
        <div key={run.id} className="py-3">
          <p className="text-sm font-medium">{run.name}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {runTime(run.start_time)} ·{" "}
            {run.source === "traces" ? `${run.span_count} ${run.span_count === 1 ? "step" : "steps"}` : "LLM request"}
          </p>
        </div>
      ))}
    </div>
  );
}

export function MatchingActivityPreview({
  offset,
  onPage,
  onSelect,
  selectedIds,
  manualSelection,
  selectedCount,
  title,
  windowLabel,
  ready,
  error,
  data,
  onOpen,
  onRetry,
}: {
  offset: number;
  onPage: (offset: number) => void;
  onSelect: (id: string, checked: boolean) => void;
  selectedIds: string[];
  manualSelection: boolean;
  selectedCount: number;
  title: string;
  windowLabel: string;
  ready: boolean;
  error: Error | null;
  data: Sample | undefined;
  onRetry: () => void;
  onOpen: (run: Sample["executions"][number]) => void;
}) {
  const paginated = data?.next_offset != null || offset > 0;
  const showSelection = selectedCount !== data?.eligible || paginated;
  const selectionData = ready && showSelection ? data : undefined;
  return (
    <section aria-label="Matching activity" className="self-start rounded-lg border">
      <div className="border-b px-4 py-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-sm font-medium" role="status">
            {title}
          </p>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Refresh matching activity"
            onClick={onRetry}
            disabled={!ready}
          >
            <RotateCw className="size-3" />
          </Button>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{windowLabel} · No analysis cost</p>
      </div>
      <div className="max-h-64 overflow-y-auto px-4">
        {ready && error && (
          <p role="alert" className="py-3 text-sm text-destructive">
            {error.message}{" "}
            <Button variant="link" onClick={onRetry}>
              Retry preview
            </Button>
          </p>
        )}
        {ready && data?.eligible === 0 && (
          <p className="py-4 text-sm text-muted-foreground">
            No matches. Try removing a condition or check that your agent records this metadata. Recent trace updates
            need two minutes to settle.
          </p>
        )}
        {ready &&
          data?.executions.map((run) => (
            <div key={run.id} className="flex items-center justify-between gap-3 border-b last:border-0">
              {manualSelection && (
                <input
                  type="checkbox"
                  aria-label={`Select ${run.name}`}
                  checked={selectedIds.includes(run.id)}
                  onChange={(e) => onSelect(run.id, e.target.checked)}
                />
              )}
              <div className="min-w-0">
                <RunList executions={[run]} />
              </div>
              {run.source === "traces" && (
                <Button variant="ghost" size="icon-sm" aria-label={`Open ${run.name}`} onClick={() => onOpen(run)}>
                  <ChevronRight className="size-4" />
                </Button>
              )}
            </div>
          ))}
      </div>
      {selectionData && (
        <div className="border-t px-4 py-3 space-y-2">
          <p className="text-xs text-muted-foreground">
            {selectedCount} selected for analysis
            {paginated && (
              <>
                {" "}
                · Showing {offset + (selectionData.executions.length ? 1 : 0)}–
                {offset + selectionData.executions.length} of {selectionData.eligible}
              </>
            )}
          </p>
          {paginated && (
            <div className="flex justify-between">
              <Button
                size="sm"
                variant="ghost"
                disabled={offset === 0}
                onClick={() => onPage(Math.max(0, offset - 100))}
              >
                Previous
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={selectionData.next_offset == null}
                onClick={() => onPage(selectionData.next_offset ?? offset)}
              >
                Next
              </Button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
