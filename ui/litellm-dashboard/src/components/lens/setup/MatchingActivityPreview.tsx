"use client";

import { memo, useEffect, useState, type ComponentProps } from "react";
import { ChevronRight } from "lucide-react";
import { useInView } from "react-intersection-observer";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cva.config";

import { runTime } from "../model/format";
import type { Execution, MatchingPreview, PreviewSelection } from "./useMatchingActivity";

const PREFETCH_MARGIN = "0px 0px 240px 0px";
const SKELETON_ROWS = 6;

type RunRowProps = ComponentProps<"div"> & { run: Execution };

function RunRow({ run, className, ...props }: RunRowProps) {
  const steps = `${run.span_count} ${run.span_count === 1 ? "step" : "steps"}`;
  return (
    <div data-slot="run-row" className={cn("min-w-0 flex-1 py-3", className)} {...props}>
      <p className="truncate text-sm font-medium">{run.name}</p>
      <p className="mt-1 text-xs text-muted-foreground">
        {runTime(run.start_time)} · {run.source === "traces" ? steps : "LLM request"}
      </p>
    </div>
  );
}

function PlaceholderRows() {
  return Array.from({ length: SKELETON_ROWS }, (_, index) => (
    <div key={index} data-testid="runs-placeholder" className="grid gap-2 border-b py-3 last:border-0">
      <Skeleton className="h-4 w-2/3" />
      <Skeleton className="h-3 w-1/3" />
    </div>
  ));
}

interface PreviewRowsProps {
  readonly executions: readonly Execution[];
  readonly selection: PreviewSelection | null;
  readonly onOpen: (run: Execution) => void;
}

/** The preview can hold hundreds of rows; they re-render only when the rows or picks change, not on every keystroke. */
const PreviewRows = memo(function PreviewRows({ executions, selection, onOpen }: PreviewRowsProps) {
  return executions.map((run) => (
    <div key={run.id} className="flex items-center justify-between gap-3 border-b last:border-0">
      {selection && (
        <input
          type="checkbox"
          aria-label={`Select ${run.name}`}
          checked={selection.ids.includes(run.id)}
          onChange={(e) => selection.toggle(run.id, e.target.checked)}
        />
      )}
      <RunRow run={run} />
      {run.source === "traces" && (
        <Button variant="ghost" size="icon-sm" aria-label={`Open ${run.name}`} onClick={() => onOpen(run)}>
          <ChevronRight className="size-4" />
        </Button>
      )}
    </div>
  ));
});

/** Runs saved by hand on an older investigation; picks are no longer made here, only cleared. */
function PickedRuns({ selection }: { selection: PreviewSelection }) {
  const picked = selection.ids.length;
  if (picked === 0) return null;
  return (
    <div className="flex items-center justify-between gap-3 border-t bg-muted/30 px-4 py-2">
      <p className="text-xs tabular-nums text-muted-foreground">
        {selection.count.toLocaleString()} selected for analysis
      </p>
      <Button variant="ghost" size="xs" onClick={selection.clear}>
        Clear {picked} selected runs
      </Button>
    </div>
  );
}

export type MatchingActivityPreviewProps = ComponentProps<"section"> &
  MatchingPreview & {
    onOpen: (run: Execution) => void;
  };

export function MatchingActivityPreview({
  status,
  page,
  selection,
  onOpen,
  className,
  ...props
}: MatchingActivityPreviewProps) {
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null);
  const { ref: tailRef, inView: nearTail } = useInView({ root: scroller, rootMargin: PREFETCH_MARGIN });
  const { loadMore, loadingMore } = page;
  const canContinue = status.ready && page.hasMore && page.executions.length > 0;
  useEffect(() => {
    if (nearTail && canContinue && !loadingMore) loadMore();
  }, [nearTail, canContinue, loadingMore, loadMore]);
  const shown = status.ready || status.stale;
  const showRows = shown && !status.error && page.executions.length > 0;
  return (
    <section
      aria-label="Matching activity"
      data-slot="matching-activity-preview"
      className={cn("flex flex-col self-start overflow-hidden rounded-lg border bg-card", className)}
      {...props}
    >
      {status.notice && <p className="px-4 py-3 text-sm text-muted-foreground">{status.notice}</p>}
      {status.ready && status.error && (
        <p role="alert" className="px-4 py-3 text-sm text-destructive">
          {status.error.message}{" "}
          <Button variant="link" onClick={status.refresh}>
            Retry preview
          </Button>
        </p>
      )}
      {status.ready && page.eligible === 0 && (
        <div className="grid gap-1 px-4 py-10 text-center">
          <p className="text-sm font-medium">No matches</p>
          <p className="text-xs text-muted-foreground">
            Try removing a condition or check that your agent records this metadata. Recent trace updates need two
            minutes to settle.
          </p>
        </div>
      )}
      {(status.loading || showRows) && (
        <div
          ref={setScroller}
          aria-busy={status.loading || status.stale || loadingMore}
          className={cn(
            "max-h-[calc(100dvh-16rem)] min-h-0 flex-1 overflow-y-auto px-4 transition-opacity lg:max-h-none",
            status.stale && "opacity-60",
          )}
        >
          {status.loading ? (
            <PlaceholderRows />
          ) : (
            <PreviewRows executions={page.executions} selection={selection} onOpen={onOpen} />
          )}
          {canContinue && (
            <p ref={tailRef} data-testid="preview-placeholder" className="py-3 text-xs text-muted-foreground">
              Loading more…
            </p>
          )}
        </div>
      )}
      {shown && selection && <PickedRuns selection={selection} />}
    </section>
  );
}
