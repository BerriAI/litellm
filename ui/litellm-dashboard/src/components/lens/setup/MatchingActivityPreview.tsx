"use client";

import { useEffect, useState, type ComponentProps } from "react";
import { ChevronRight, RotateCw } from "lucide-react";
import { useInView } from "react-intersection-observer";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { runTime } from "../model/format";
import type { Execution, MatchingPreview } from "./useMatchingActivity";

const PREFETCH_MARGIN = "0px 0px 240px 0px";

type RunRowProps = ComponentProps<"div"> & { run: Execution };

function RunRow({ run, className, ...props }: RunRowProps) {
  const steps = `${run.span_count} ${run.span_count === 1 ? "step" : "steps"}`;
  return (
    <div data-slot="run-row" className={cn("min-w-0 py-3", className)} {...props}>
      <p className="text-sm font-medium">{run.name}</p>
      <p className="mt-1 text-xs text-muted-foreground">
        {runTime(run.start_time)} · {run.source === "traces" ? steps : "LLM request"}
      </p>
    </div>
  );
}

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
  return (
    <section
      aria-label="Matching activity"
      data-slot="matching-activity-preview"
      className={cn("self-start rounded-lg border", className)}
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
      <div ref={setScroller} aria-busy={loadingMore} className="max-h-[60dvh] overflow-y-auto px-4">
        {status.ready && status.error && (
          <p role="alert" className="py-3 text-sm text-destructive">
            {status.error.message}{" "}
            <Button variant="link" onClick={status.refresh}>
              Retry preview
            </Button>
          </p>
        )}
        {status.ready && page.eligible === 0 && (
          <p className="py-4 text-sm text-muted-foreground">
            No matches. Try removing a condition or check that your agent records this metadata. Recent trace updates
            need two minutes to settle.
          </p>
        )}
        {status.ready &&
          page.executions.map((run) => (
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
          ))}
        {canContinue && (
          <p ref={tailRef} data-testid="preview-placeholder" className="py-3 text-xs text-muted-foreground">
            Loading more…
          </p>
        )}
      </div>
      {status.ready && <PreviewFooter page={page} selection={selection} />}
    </section>
  );
}
