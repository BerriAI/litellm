"use client";

import { Bot, Check, ChevronsUpDown, Search } from "lucide-react";
import { useState } from "react";

import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/cva.config";

import { FrameworkLogo, traceFramework } from "../traces/ui/TraceFramework";
import type { AgentSummary } from "./agentRollup";

export function AgentMark({ agent }: { agent: Pick<AgentSummary, "frameworks"> | undefined }) {
  const framework = agent ? traceFramework({ frameworks: [...agent.frameworks] }) : null;
  return framework ? (
    <FrameworkLogo framework={framework} className="size-4" />
  ) : (
    <Bot aria-hidden className="size-4 shrink-0 text-muted-foreground" />
  );
}

export const matchesAgent = (agent: Pick<AgentSummary, "name">, query: string): boolean =>
  agent.name.toLowerCase().includes(query.trim().toLowerCase());

interface AgentPickerProps {
  agent: string;
  agents: readonly AgentSummary[];
  onSelect: (agent: string) => void;
}

const ITEM = "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted";

/** The agent every Lens view is scoped to, like the project switcher in Braintrust. */
export function AgentPicker({ agent, agents, onSelect }: AgentPickerProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const shown = agents.filter((item) => matchesAgent(item, query));
  const choose = (next: string) => {
    setOpen(false);
    setQuery("");
    onSelect(next);
  };
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        aria-label={`Agent: ${agent}`}
        className="inline-flex h-7 max-w-56 items-center gap-1.5 rounded-md border px-2 text-sm font-medium hover:bg-muted"
      >
        <AgentMark agent={agents.find((item) => item.name === agent)} />
        <span className="truncate">{agent}</span>
        <ChevronsUpDown aria-hidden className="size-3.5 shrink-0 text-muted-foreground" />
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72 gap-1 p-1.5">
        <div className="relative">
          <Search aria-hidden className="absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            aria-label="Find agent"
            placeholder="Find agent"
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="h-8 border-0 pl-8 text-sm shadow-none focus-visible:ring-0"
          />
        </div>
        <div className="border-t" />
        <p className="px-2 pt-1.5 pb-0.5 text-xs text-muted-foreground">Agents</p>
        <ul aria-label="Agents" className="flex max-h-72 flex-col overflow-y-auto">
          {shown.map((item) => (
            <li key={item.name}>
              <button type="button" className={ITEM} onClick={() => choose(item.name)}>
                <Check aria-hidden className={cn("size-3.5 shrink-0", item.name !== agent && "invisible")} />
                <AgentMark agent={item} />
                <span className="truncate">{item.name}</span>
                <span className="ml-auto text-xs text-muted-foreground tabular-nums">{item.runs.toLocaleString()}</span>
              </button>
            </li>
          ))}
          {shown.length === 0 && <li className="px-2 py-1.5 text-sm text-muted-foreground">No agents match</li>}
        </ul>
      </PopoverContent>
    </Popover>
  );
}
