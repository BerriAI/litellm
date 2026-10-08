"use client";

import type { TraceSummary } from "../../types";
import type { TimeWindow } from "@/components/shared/timeRange/timeRange";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useRunFilterRouting } from "../../routing";
import { RunSearch } from "./RunSearch";

interface RunsToolbarProps {
  query: string;
  onQueryChange: (value: string) => void;
  runs: readonly TraceSummary[];
  /** The range the list shows, for the copied query. */
  range?: TimeWindow;
  /** The list is reloading for a new range; the search box shows a spinner. */
  busy?: boolean;
  /** Extra controls (time range, live tail) rendered on the right. */
  children?: React.ReactNode;
}

export function RunsToolbar({ query, onQueryChange, runs, range, busy, children }: RunsToolbarProps) {
  const { status, setStatus } = useRunFilterRouting();
  const statuses = [
    { value: "all", label: "All status" },
    { value: "ok", label: "No errors" },
    { value: "error", label: "With errors" },
  ];
  return (
    <div className="flex shrink-0 flex-col bg-card">
      <div className="flex min-h-10 flex-wrap items-stretch border-b border-border">
        <RunSearch value={query} onChange={onQueryChange} runs={runs} range={range} busy={busy} />
        {children && <div className="ml-auto flex h-10 max-w-full items-stretch">{children}</div>}
      </div>
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <Select
          items={statuses}
          value={status}
          onValueChange={(value: "all" | "ok" | "error" | null) => value !== null && setStatus(value)}
        >
          <SelectTrigger size="sm" className="h-8 min-w-28 text-xs" aria-label="Filter traces by status">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {statuses.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
