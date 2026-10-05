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
    <div className="flex h-10 shrink-0 items-stretch border-b border-border bg-card">
      <RunSearch value={query} onChange={onQueryChange} runs={runs} range={range} />
      {children && <div className="flex shrink-0 items-stretch">{children}</div>}
    </div>
  );
}
