"use client";

import type { TraceSummary } from "../../types";
import type { TimeWindow } from "@/components/shared/timeline/Timeline";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useRunFilterRouting } from "../../routing";
import { traceAgentNames } from "../../utils";
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
  const { agent, status, setAgent, setStatus } = useRunFilterRouting();
  const agents = [
    { value: "", label: "All agents" },
    ...[...new Set([...runs.flatMap(traceAgentNames), ...(agent ? [agent] : [])])]
      .sort()
      .map((name) => ({ value: name, label: name })),
  ];
  const statuses = [
    { value: "all", label: "All status" },
    { value: "ok", label: "No errors" },
    { value: "error", label: "With errors" },
  ];
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-2 border-b bg-card p-2">
      <RunSearch value={query} onChange={onQueryChange} runs={runs} range={range} />
      <Select items={agents} value={agent} onValueChange={(value) => value !== null && setAgent(value)}>
        <SelectTrigger size="sm" className="h-8 max-w-48 min-w-32 text-xs" aria-label="Filter traces by agent">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {agents.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
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
      {children && <div className="ml-auto flex max-w-full flex-wrap items-center justify-end gap-2">{children}</div>}
    </div>
  );
}
