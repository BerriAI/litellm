import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import { getAgentsList } from "../networking";

interface Agent {
  agent_id: string;
  agent_name: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  agent_config?: Record<string, any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  agent_card_params?: Record<string, any>;
}

interface AgentSelectorProps {
  onChange: (selected: { agents: string[]; accessGroups: string[] }) => void;
  value?: { agents: string[]; accessGroups: string[] };
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
}

const AgentSelector: React.FC<AgentSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select agents",
  disabled = false,
}) => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [accessGroups, setAccessGroups] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      if (!accessToken) return;
      try {
        const response = await getAgentsList(accessToken);
        const agentsList = response?.agents || [];
        setAgents(agentsList);

        const groups = new Set<string>();
        agentsList.forEach((agent: Agent) => {
          const agentAccessGroups =
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (agent as any).agent_access_groups as string[] | undefined;
          if (agentAccessGroups && Array.isArray(agentAccessGroups)) {
            agentAccessGroups.forEach((g: string) => groups.add(g));
          }
        });
        setAccessGroups(Array.from(groups));
      } catch (error) {
        console.error("Error fetching agents:", error);
      }
    };
    fetchData();
  }, [accessToken]);

  const options = useMemo(
    () => [
      ...accessGroups.map((group) => ({
        label: group,
        value: `group:${group}`,
        isAccessGroup: true,
        searchText: `${group} Access Group`,
      })),
      ...agents.map((agent) => ({
        label: `${agent.agent_name || agent.agent_id}`,
        value: agent.agent_id,
        isAccessGroup: false,
        searchText: `${agent.agent_name || agent.agent_id} ${agent.agent_id} Agent`,
      })),
    ],
    [accessGroups, agents],
  );

  const selectedValues = useMemo(
    () => [
      ...(value?.agents || []),
      ...(value?.accessGroups || []).map((g) => `group:${g}`),
    ],
    [value],
  );

  const filteredOptions = useMemo(
    () =>
      options
        .filter((o) => !selectedValues.includes(o.value))
        .filter((o) =>
          query
            ? o.searchText.toLowerCase().includes(query.toLowerCase())
            : true,
        ),
    [options, selectedValues, query],
  );

  const labelFor = (v: string) =>
    options.find((o) => o.value === v)?.label ?? v;
  const isGroupFor = (v: string) =>
    options.find((o) => o.value === v)?.isAccessGroup ?? false;

  const applyChange = (next: string[]) => {
    const agentsSelected = next.filter((v) => !v.startsWith("group:"));
    const accessGroupsSelected = next
      .filter((v) => v.startsWith("group:"))
      .map((v) => v.replace("group:", ""));
    onChange({ agents: agentsSelected, accessGroups: accessGroupsSelected });
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "min-h-9 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-left disabled:opacity-50",
            className,
          )}
        >
          {selectedValues.length === 0 ? (
            <span className="text-muted-foreground px-1">{placeholder}</span>
          ) : (
            selectedValues.map((v) => (
              <Badge
                key={v}
                variant="secondary"
                className="gap-1 inline-flex items-center"
              >
                <span
                  className={cn(
                    "w-1.5 h-1.5 rounded-full inline-block",
                    isGroupFor(v) ? "bg-emerald-500" : "bg-purple-500",
                  )}
                />
                {labelFor(v)}
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => {
                    e.stopPropagation();
                    applyChange(selectedValues.filter((s) => s !== v));
                  }}
                  className="inline-flex items-center"
                  aria-label={`Remove ${labelFor(v)}`}
                >
                  <X size={12} />
                </span>
              </Badge>
            ))
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[var(--radix-popover-trigger-width)] p-2"
      >
        <Input
          autoFocus
          placeholder="Search agents, groups…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="h-8 mb-2"
        />
        <div className="max-h-60 overflow-y-auto">
          {filteredOptions.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No matches
            </div>
          ) : (
            filteredOptions.map((opt) => (
              <button
                key={opt.value}
                type="button"
                className="w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent flex items-center gap-2"
                onClick={() => applyChange([...selectedValues, opt.value])}
              >
                <span
                  className={cn(
                    "w-2 h-2 rounded-full inline-block shrink-0",
                    opt.isAccessGroup ? "bg-emerald-500" : "bg-purple-500",
                  )}
                />
                <span className="flex-1 truncate">{opt.label}</span>
                <span
                  className={cn(
                    "text-xs font-medium opacity-80",
                    opt.isAccessGroup
                      ? "text-emerald-700 dark:text-emerald-400"
                      : "text-purple-700 dark:text-purple-400",
                  )}
                >
                  {opt.isAccessGroup ? "Access Group" : "Agent"}
                </span>
              </button>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default AgentSelector;
