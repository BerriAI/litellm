import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import React, { useMemo, useState } from "react";

interface MCPServerSelectorProps {
  onChange: (selected: {
    servers: string[];
    accessGroups: string[];
    toolsets: string[];
  }) => void;
  value?: {
    servers: string[];
    accessGroups: string[];
    toolsets?: string[];
  };
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
  teamId?: string | null;
}

const TOOLSET_PREFIX = "toolset:";

/**
 * Multi-select that combines three lists:
 *  - MCP Access Groups (emerald dot)
 *  - MCP Servers       (blue dot)
 *  - Toolsets          (purple dot)
 *
 * Selected items become chips; popover shows the three categories together
 * with an inline search input.
 */
const MCPServerSelector: React.FC<MCPServerSelectorProps> = ({
  onChange,
  value,
  className,
  placeholder = "Select MCP servers",
  disabled = false,
  teamId,
}) => {
  const { data: mcpServers = [] } = useMCPServers(teamId);
  const { data: accessGroups = [] } = useMCPAccessGroups();
  const { data: toolsets = [] } = useMCPToolsets();

  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const accessGroupSet = useMemo(
    () => new Set(accessGroups),
    [accessGroups],
  );

  const options = useMemo(
    () => [
      ...accessGroups.map((group) => ({
        label: group,
        value: group,
        type: "accessGroup" as const,
        searchText: `${group} Access Group`,
      })),
      ...mcpServers.map((server) => ({
        label: `${server.server_name || server.server_id} (${server.server_id})`,
        value: server.server_id,
        type: "server" as const,
        searchText: `${server.server_name || server.server_id} ${server.server_id} MCP Server`,
      })),
      ...toolsets.map((toolset) => ({
        label: toolset.toolset_name,
        value: `${TOOLSET_PREFIX}${toolset.toolset_id}`,
        type: "toolset" as const,
        searchText: `${toolset.toolset_name} ${toolset.toolset_id} Toolset`,
      })),
    ],
    [accessGroups, mcpServers, toolsets],
  );

  const selectedValues = useMemo(
    () => [
      ...(value?.servers || []),
      ...(value?.accessGroups || []),
      ...(value?.toolsets || []).map((id) => `${TOOLSET_PREFIX}${id}`),
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

  const dotByType: Record<string, string> = {
    accessGroup: "bg-emerald-500",
    server: "bg-blue-500",
    toolset: "bg-purple-500",
  };

  const labelByType: Record<string, string> = {
    accessGroup: "Access Group",
    server: "MCP Server",
    toolset: "Toolset",
  };

  const textByType: Record<string, string> = {
    accessGroup: "text-emerald-700 dark:text-emerald-400",
    server: "text-blue-700 dark:text-blue-400",
    toolset: "text-purple-700 dark:text-purple-400",
  };

  const applyChange = (next: string[]) => {
    const toolsetsSelected = next
      .filter((v) => v.startsWith(TOOLSET_PREFIX))
      .map((v) => v.slice(TOOLSET_PREFIX.length));
    const rest = next.filter((v) => !v.startsWith(TOOLSET_PREFIX));
    const servers = rest.filter((v) => !accessGroupSet.has(v));
    const accessGroupsSelected = rest.filter((v) => accessGroupSet.has(v));
    onChange({
      servers,
      accessGroups: accessGroupsSelected,
      toolsets: toolsetsSelected,
    });
  };

  const labelFor = (v: string) => {
    const opt = options.find((o) => o.value === v);
    return opt ? opt.label : v;
  };
  const typeFor = (v: string) => {
    const opt = options.find((o) => o.value === v);
    return opt?.type ?? "server";
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
                    dotByType[typeFor(v)],
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
          placeholder="Search servers, groups, toolsets…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="h-8 mb-2"
        />
        <div className="max-h-72 overflow-y-auto">
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
                onClick={() =>
                  applyChange([...selectedValues, opt.value])
                }
              >
                <span
                  className={cn(
                    "w-2 h-2 rounded-full inline-block shrink-0",
                    dotByType[opt.type],
                  )}
                />
                <span className="flex-1 truncate">{opt.label}</span>
                <span
                  className={cn(
                    "text-xs font-medium opacity-80",
                    textByType[opt.type],
                  )}
                >
                  {labelByType[opt.type]}
                </span>
              </button>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default MCPServerSelector;
