"use client";

import { ChevronRight } from "lucide-react";
import { useState } from "react";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useNow } from "@/hooks/useNow";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import { agoLabel } from "@/components/view_logs/TraceView/lensField";
import { cn } from "@/lib/cva.config";

import { ALL_AGENTS, filterInbox, inboxAgents, inboxRows, type InboxRow, type Priority } from "../model/inbox";
import type { Lens } from "../model/types";

const PRIORITY_DOT = { high: "bg-[#e5484d]", medium: "bg-amber-500", low: "bg-muted-foreground/50" } as const;
const PRIORITY_ITEMS: { value: Priority | "all"; label: string }[] = [
  { value: "all", label: "All priorities" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
];

const TH = "px-3 font-medium";
const TH_NUM = "px-3 text-right font-medium";

function FilterSelect<T extends string>({
  label,
  value,
  items,
  onChange,
  width,
}: {
  label: string;
  value: T;
  items: { value: T; label: string }[];
  onChange: (value: T) => void;
  width: string;
}) {
  return (
    <Select items={items} value={value} onValueChange={(next: T | null) => next !== null && onChange(next)}>
      <SelectTrigger size="sm" className={cn("h-7 text-[12px]", width)} aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function FindingsInbox({ lenses, onOpen }: { lenses: readonly Lens[]; onOpen: (row: InboxRow) => void }) {
  const [agent, setAgent] = useState(ALL_AGENTS);
  const [priority, setPriority] = useState<Priority | "all">("all");
  const now = useNow(30000);
  const all = inboxRows(lenses);
  const rows = filterInbox(all, { agent, priority });
  const agentItems = [
    { value: ALL_AGENTS, label: "All agents" },
    ...inboxAgents(all).map((a) => ({ value: a, label: a })),
  ];
  return (
    <div
      className="flex min-h-[420px] flex-1 flex-col overflow-hidden border-y border-border bg-card"
      data-testid="findings-inbox"
    >
      <div className="flex min-h-10 shrink-0 flex-wrap items-center gap-2 border-b border-border bg-card px-3 py-1.5">
        <FilterSelect
          label="Filter by agent"
          value={agent}
          items={agentItems}
          onChange={setAgent}
          width="min-w-[150px]"
        />
        <FilterSelect
          label="Filter by priority"
          value={priority}
          items={PRIORITY_ITEMS}
          onChange={setPriority}
          width="min-w-[130px]"
        />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <table aria-label="Findings" className="w-full min-w-[900px] table-fixed border-collapse text-left">
          <thead className="sticky top-0 z-sticky bg-muted/40 backdrop-blur">
            <tr className="h-8 border-b border-border text-[10px] tracking-[0.08em] text-muted-foreground uppercase">
              <th className={`w-[96px] ${TH}`}>Priority</th>
              <th className={TH}>Finding</th>
              <th className={`w-[180px] ${TH}`}>Agent</th>
              <th className={`w-[64px] ${TH_NUM}`}>Runs</th>
              <th className={`w-[96px] ${TH}`}>Last seen</th>
              <th className="w-8" />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.key}
                onClick={() => onOpen(row)}
                data-testid="inbox-row"
                aria-label={row.title}
                className="h-9 cursor-pointer border-b border-border/60 text-[12px] transition-colors duration-150 hover:bg-trace-row-hover motion-reduce:transition-none"
              >
                <td className="px-3">
                  <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                    <span className={cn("size-1.5 rounded-full", PRIORITY_DOT[row.priority])} />
                    {row.priority}
                  </span>
                </td>
                <td className="px-3" title={row.suggestion ? `Fix: ${row.suggestion}` : undefined}>
                  <span className="block truncate text-foreground">{row.title}</span>
                </td>
                <td className="truncate px-3 text-muted-foreground" title={row.agents.join(", ")}>
                  {row.agents.join(", ")}
                </td>
                <td className="px-3 text-right font-mono tabular-nums text-foreground">{row.runs}</td>
                <td
                  className="px-3 font-mono text-[11px] tabular-nums text-muted-foreground"
                  title={formatActivityTimestamp(row.lastSeen)}
                >
                  {agoLabel(Date.parse(row.lastSeen), now)}
                </td>
                <td>
                  <ChevronRight className="size-3 text-muted-foreground/60" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && (
          <div className="py-16 text-center text-[12px] text-muted-foreground">
            {all.length === 0
              ? "No open findings yet. New problems show up here as soon as an investigation spots them."
              : "No findings match these filters."}
          </div>
        )}
      </div>
      <footer className="flex h-8 shrink-0 items-center border-t border-border bg-muted/40 px-3 font-mono text-[11px] text-muted-foreground">
        {rows.length} {rows.length === 1 ? "finding" : "findings"}
        {rows.length !== all.length && ` of ${all.length}`}
      </footer>
    </div>
  );
}
