"use client";

import type { TraceSummary } from "../../types";
import type { TimeWindow } from "@/components/shared/timeline/Timeline";

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
    <div className="@container/traces shrink-0 border-b border-border bg-card">
      <div className="grid grid-cols-1 @4xl/traces:grid-cols-[minmax(0,1fr)_auto]">
        <RunSearch value={query} onChange={onQueryChange} runs={runs} range={range} />
        {children && (
          <div className="flex min-h-11 min-w-0 items-stretch justify-end border-t border-border @4xl/traces:border-t-0">
            {children}
          </div>
        )}
      </div>
    </div>
  );
}
