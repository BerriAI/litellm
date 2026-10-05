"use client";

import { type ComponentProps } from "react";
import { RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { AgentTracesTable, type RunPicks } from "../traces/list/AgentTracesTable";
import type { TraceSummary } from "../traces/types";
import type { MatchingPreview, PreviewSelection } from "./useMatchingActivity";

const executionOf = (run: TraceSummary) => `${run.trace_ref}:${run.trace_id}`;

const runPicks = (selection: PreviewSelection): RunPicks => ({
  isPicked: (run) => selection.ids.includes(executionOf(run)),
  toggle: (run, picked) => selection.toggle(executionOf(run), picked),
});

type PreviewFooterProps = ComponentProps<"div"> & Pick<MatchingPreview, "page" | "selection">;

/** Selection count and a way to undo manual picks; hidden while every match is simply going to be analyzed. */
function PreviewFooter({ page, selection, className, ...props }: PreviewFooterProps) {
  const partial = page.eligible != null && page.executions.length < page.eligible;
  const count = selection?.count ?? page.selected;
  const picked = selection?.ids.length ?? 0;
  const everything = count === page.eligible && !partial && picked === 0;
  if (page.eligible == null || everything) return null;
  return (
    <div
      data-slot="preview-footer"
      className={cn("flex flex-wrap items-center justify-between gap-3 border-t px-4 py-3", className)}
      {...props}
    >
      <p className="text-xs text-muted-foreground">
        {count} selected for analysis
        {partial && (
          <>
            {" "}
            · Showing {page.executions.length} of {page.eligible}
          </>
        )}
      </p>
      {selection && picked > 0 && (
        <Button variant="outline" size="sm" onClick={selection.clear}>
          Clear {picked} selected runs
        </Button>
      )}
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
  const runs = page.executions.flatMap((run) => (run.summary ? [run.summary] : []));
  const shown = status.ready || status.stale;
  return (
    <section
      aria-label="Matching activity"
      data-slot="matching-activity-preview"
      className={cn("self-start overflow-hidden rounded-lg border", className)}
      {...props}
    >
      <div className="border-b px-4 py-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-sm font-medium" role="status">
            {status.title}
          </p>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Refresh matching activity"
            onClick={status.refresh}
            disabled={!status.ready}
          >
            <RotateCw className="size-3" />
          </Button>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{status.windowLabel} · No analysis cost</p>
      </div>
      {status.ready && status.error && (
        <p role="alert" className="px-4 py-3 text-sm text-destructive">
          {status.error.message}{" "}
          <Button variant="link" onClick={status.refresh}>
            Retry preview
          </Button>
        </p>
      )}
      {status.ready && page.eligible === 0 && (
        <p className="px-4 py-4 text-sm text-muted-foreground">
          No matches. Try removing a filter from the search. Recent trace updates need two minutes to settle.
        </p>
      )}
      {(status.loading || (shown && !status.error && runs.length > 0)) && (
        <div className="max-h-[60dvh] min-h-0 overflow-auto">
          <AgentTracesTable
            traces={runs}
            isLoading={status.loading}
            error={null}
            hasMore={page.hasMore}
            isFetching={page.loadingMore}
            isPlaceholder={status.stale}
            onLoadMore={page.loadMore}
            onSetUpTracing={() => {}}
            picks={selection ? runPicks(selection) : undefined}
          />
        </div>
      )}
      {shown && <PreviewFooter page={page} selection={selection} />}
    </section>
  );
}
