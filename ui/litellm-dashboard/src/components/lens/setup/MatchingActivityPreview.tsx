"use client";

import { useEffect, useState } from "react";
import { ChevronRight, RotateCw } from "lucide-react";
import { useInView } from "react-intersection-observer";
import { Button } from "@/components/ui/button";

import { type Sample } from "../model/types";
import { runTime } from "../model/format";

const PREFETCH_MARGIN = "0px 0px 240px 0px";

type Execution = Sample["executions"][number];

export function RunList({ executions }: { executions: Execution[] }) {
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
  onSelect,
  selectedIds,
  manualSelection,
  selectedCount,
  title,
  windowLabel,
  ready,
  error,
  eligible,
  executions,
  hasMore,
  loadingMore,
  onLoadMore,
  onOpen,
  onRetry,
}: {
  onSelect: (id: string, checked: boolean) => void;
  selectedIds: string[];
  manualSelection: boolean;
  selectedCount: number;
  title: string;
  windowLabel: string;
  ready: boolean;
  error: Error | null;
  eligible: number | undefined;
  executions: Execution[];
  hasMore: boolean;
  loadingMore: boolean;
  onLoadMore: () => void;
  onRetry: () => void;
  onOpen: (run: Execution) => void;
}) {
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null);
  const { ref: tailRef, inView: nearTail } = useInView({ root: scroller, rootMargin: PREFETCH_MARGIN });
  const canContinue = ready && hasMore && executions.length > 0;
  useEffect(() => {
    if (nearTail && canContinue && !loadingMore) onLoadMore();
  }, [nearTail, canContinue, loadingMore, onLoadMore]);
  const partial = eligible != null && executions.length < eligible;
  const showSelection = ready && eligible != null && (selectedCount !== eligible || partial);
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
      <div ref={setScroller} aria-busy={loadingMore} className="max-h-[60dvh] overflow-y-auto px-4">
        {ready && error && (
          <p role="alert" className="py-3 text-sm text-destructive">
            {error.message}{" "}
            <Button variant="link" onClick={onRetry}>
              Retry preview
            </Button>
          </p>
        )}
        {ready && eligible === 0 && (
          <p className="py-4 text-sm text-muted-foreground">
            No matches. Try removing a condition or check that your agent records this metadata. Recent trace updates
            need two minutes to settle.
          </p>
        )}
        {ready &&
          executions.map((run) => (
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
        {canContinue && (
          <p ref={tailRef} data-testid="preview-placeholder" className="py-3 text-xs text-muted-foreground">
            Loading more…
          </p>
        )}
      </div>
      {showSelection && (
        <div className="border-t px-4 py-3">
          <p className="text-xs text-muted-foreground">
            {selectedCount} selected for analysis
            {partial && (
              <>
                {" "}
                · Showing {executions.length} of {eligible}
              </>
            )}
          </p>
        </div>
      )}
    </section>
  );
}
