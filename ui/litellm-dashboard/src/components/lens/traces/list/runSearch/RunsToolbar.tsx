"use client";

import type { TraceSummary } from "../../types";
import type { TimeWindow } from "../TracesTimeline";

import { RunSearch } from "./RunSearch";

interface RunsToolbarProps {
  query: string;
  onQueryChange: (value: string) => void;
  runs: readonly TraceSummary[];
  /** The range the list shows, for the copied query. */
  range?: TimeWindow;
  /** Extra controls (time range, live tail) rendered on the right. */
  children?: React.ReactNode;
}

export function RunsToolbar({ query, onQueryChange, runs, range, children }: RunsToolbarProps) {
  return (
    <div className="flex min-h-11 shrink-0 flex-wrap items-center gap-2 border-b border-border bg-card p-2">
      <RunSearch value={query} onChange={onQueryChange} runs={runs} range={range} />
      {children && <div className="ml-auto flex items-center gap-2">{children}</div>}
    </div>
  );
}
