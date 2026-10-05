"use client";

import type { TraceSummary } from "../../types";
import type { TimeWindow } from "@/components/shared/timeRange/timeRange";

import type { RunOrder } from "../runOrder";
import { RunSearch } from "./RunSearch";

interface RunsToolbarProps {
  query: string;
  onQueryChange: (value: string) => void;
  runs: readonly TraceSummary[];
  /** The range and order the list shows, for the copied query. */
  range?: TimeWindow;
  order?: RunOrder;
  /** Extra controls (time range, live tail) rendered on the right. */
  children?: React.ReactNode;
}

export function RunsToolbar({ query, onQueryChange, runs, range, order, children }: RunsToolbarProps) {
  return (
    <div className="flex h-10 shrink-0 items-stretch border-b border-border bg-card">
      <RunSearch value={query} onChange={onQueryChange} runs={runs} range={range} order={order} />
      {children && <div className="flex shrink-0 items-stretch">{children}</div>}
    </div>
  );
}
