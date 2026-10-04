"use client";

import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";

import { RunSearch } from "./RunSearch";

interface RunsToolbarProps {
  query: string;
  onQueryChange: (value: string) => void;
  runs: readonly TraceSummary[];
  /** Extra controls (time range, live tail) rendered on the right. */
  children?: React.ReactNode;
}

export function RunsToolbar({ query, onQueryChange, runs, children }: RunsToolbarProps) {
  return (
    <div className="flex min-h-11 shrink-0 flex-wrap items-center gap-2 border-b border-border bg-card px-3 py-2">
      <RunSearch value={query} onChange={onQueryChange} runs={runs} />
      {children && <div className="ml-auto flex items-center gap-2">{children}</div>}
    </div>
  );
}
