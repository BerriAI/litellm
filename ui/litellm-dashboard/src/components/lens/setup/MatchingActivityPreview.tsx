"use client";

import { memo, useMemo, type ComponentProps } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { AgentTracesTable, type RunPicks } from "../traces/list/AgentTracesTable";
import type { TraceSummary } from "../traces/types";
import type { MatchingPreview, PreviewSelection } from "./useMatchingActivity";

const executionOf = (run: TraceSummary) => `${run.id}:${run.trace_id}`;

const runPicks = (ids: PreviewSelection["ids"], toggle: PreviewSelection["toggle"]): RunPicks => ({
  isPicked: (run) => ids.includes(executionOf(run)),
  toggle: (run, picked) => toggle(executionOf(run), picked),
});

const noSetup = () => {};

/** The preview can hold hundreds of rows; it re-renders only when its rows or paging change, not on every keystroke. */
const PreviewTable = memo(AgentTracesTable);

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

export type MatchingActivityPreviewProps = ComponentProps<"section"> & MatchingPreview;

/** The runs the draft selection matches, in the same table as the Traces tab. */
export function MatchingActivityPreview({
  status,
  page,
  selection,
  className,
  ...props
}: MatchingActivityPreviewProps) {
  const runs = useMemo(() => page.executions.flatMap((run) => (run.summary ? [run.summary] : [])), [page.executions]);
  const pickedIds = selection?.ids;
  const togglePick = selection?.toggle;
  const picks = useMemo(
    () => (pickedIds && togglePick ? runPicks(pickedIds, togglePick) : undefined),
    [pickedIds, togglePick],
  );
  const shown = status.ready || status.stale;
  const showTable = status.loading || (shown && !status.error && runs.length > 0);
  return (
    <section
      aria-label="Matching activity"
      data-slot="matching-activity-preview"
      className={cn("flex flex-col self-start overflow-hidden rounded-lg border bg-card", className)}
      {...props}
    >
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
            Try removing a filter from the search. Recent trace updates need two minutes to settle.
          </p>
        </div>
      )}
      {showTable && (
        <div className="flex max-h-[calc(100dvh-16rem)] min-h-0 flex-1 flex-col lg:max-h-none">
          <PreviewTable
            traces={runs}
            isLoading={status.loading}
            error={null}
            hasMore={page.hasMore}
            isFetching={page.loadingMore}
            isPlaceholder={status.stale}
            onLoadMore={page.loadMore}
            onSetUpTracing={noSetup}
            picks={picks}
          />
        </div>
      )}
      {shown && selection && <PickedRuns selection={selection} />}
    </section>
  );
}
