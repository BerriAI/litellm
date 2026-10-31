"use client";

import { Bot, Puzzle, Server } from "lucide-react";
import * as React from "react";

import type { Plugin } from "@/components/claude_code_plugins/types";
import type { AgentCard, MCPServerData } from "@/components/PublicModelHubTableColumns";
import { ItemActions, ItemContent, ItemDescription, ItemMedia } from "@/components/ui/item";

import { AgentInspector } from "./AgentInspector";
import { FilterChip } from "./FilterChip";
import { mcpAuthLabel } from "./hubFormat";
import { HubMasterDetail } from "./HubMasterDetail";
import { NoResults } from "./HubEmpty";
import { RowTitle, SearchInput } from "./HubParts";
import { McpInspector } from "./McpInspector";
import { SkillInspector } from "./SkillInspector";

const matches = (fields: (string | null | undefined)[], query: string) => {
  const needle = query.trim().toLowerCase();
  return !needle || fields.some((field) => field?.toLowerCase().includes(needle));
};

interface ResourcePaneProps<T> {
  rows: T[];
  loading: boolean;
  noun: string;
  icon: React.ReactNode;
  search: (row: T) => (string | null | undefined)[];
  filter: { label: string; noun: string; values: (row: T) => string[] };
  selected: string | null;
  onSelect: (key: string | null) => void;
  getKey: (row: T) => string;
  renderDetail: (row: T) => React.ReactNode;
  title: (row: T) => string;
  description: (row: T) => string | null | undefined;
  meta: (row: T) => string | null | undefined;
}

function ResourcePane<T>({
  rows,
  loading,
  noun,
  icon,
  search,
  filter,
  selected,
  onSelect,
  getKey,
  renderDetail,
  title,
  description,
  meta,
}: ResourcePaneProps<T>) {
  const [query, setQuery] = React.useState("");
  const [chosen, setChosen] = React.useState<string[]>([]);
  const options = [...new Set(rows.flatMap(filter.values))].sort();
  const shown = rows.filter(
    (row) => matches(search(row), query) && (chosen.length === 0 || filter.values(row).some((v) => chosen.includes(v))),
  );
  const filtered = query.trim() !== "" || chosen.length > 0;
  const clear = () => {
    setQuery("");
    setChosen([]);
  };
  return (
    <HubMasterDetail
      rows={shown}
      loading={loading}
      selectedKey={selected}
      getKey={getKey}
      onSelect={onSelect}
      empty={<NoResults icon={icon} noun={noun} query={query} onClear={filtered ? clear : undefined} />}
      toolbar={
        <div className="flex items-center gap-2">
          <SearchInput value={query} onChange={setQuery} placeholder={`Search ${noun}`} />
          {options.length > 0 && (
            <FilterChip
              label={filter.label}
              noun={filter.noun}
              options={options.map((option) => ({ value: option, label: option }))}
              value={chosen}
              onChange={setChosen}
            />
          )}
        </div>
      }
      renderItem={(row) => (
        <>
          <ItemMedia variant="icon" className="size-5 text-muted-foreground">
            {icon}
          </ItemMedia>
          <ItemContent className="min-w-0 gap-0.5">
            <RowTitle>{title(row)}</RowTitle>
            {description(row) && <ItemDescription>{description(row)}</ItemDescription>}
          </ItemContent>
          {meta(row) && <ItemActions className="text-xs text-muted-foreground">{meta(row)}</ItemActions>}
        </>
      )}
      renderDetail={renderDetail}
    />
  );
}

interface PaneProps<T> {
  rows: T[] | undefined;
  loading: boolean;
  selected: string | null;
  onSelect: (key: string | null) => void;
}

export function AgentsPane({ rows, ...props }: PaneProps<AgentCard>) {
  return (
    <ResourcePane<AgentCard>
      {...props}
      rows={rows ?? []}
      noun="agents"
      icon={<Bot />}
      search={(agent) => [agent.name, agent.description]}
      filter={{
        label: "Skill tag",
        noun: "skill tags",
        values: (agent) => (agent.skills ?? []).flatMap((s) => s.tags ?? []),
      }}
      getKey={(agent) => agent.name}
      title={(agent) => agent.name}
      description={(agent) => agent.description}
      meta={(agent) => (agent.version ? `v${agent.version}` : undefined)}
      renderDetail={(agent) => <AgentInspector agent={agent} />}
    />
  );
}

export function McpPane({ rows, ...props }: PaneProps<MCPServerData>) {
  return (
    <ResourcePane<MCPServerData>
      {...props}
      rows={rows ?? []}
      noun="MCP servers"
      icon={<Server />}
      search={(server) => [server.server_name, server.mcp_info?.description]}
      filter={{
        label: "Transport",
        noun: "transports",
        values: (server) => (server.transport ? [server.transport] : []),
      }}
      getKey={(server) => server.server_name}
      title={(server) => server.server_name}
      description={(server) => server.mcp_info?.description}
      meta={(server) => mcpAuthLabel(server.auth_type)}
      renderDetail={(server) => <McpInspector server={server} />}
    />
  );
}

export function SkillsPane({ rows, ...props }: PaneProps<Plugin>) {
  return (
    <ResourcePane<Plugin>
      {...props}
      rows={rows ?? []}
      noun="skills"
      icon={<Puzzle />}
      search={(skill) => [skill.name, skill.description, skill.namespace, skill.domain, ...(skill.keywords ?? [])]}
      filter={{ label: "Domain", noun: "domains", values: (skill) => (skill.domain ? [skill.domain] : []) }}
      getKey={(skill) => skill.name}
      title={(skill) => skill.name}
      description={(skill) => skill.description}
      meta={(skill) => skill.domain}
      renderDetail={(skill) => <SkillInspector skill={skill} />}
    />
  );
}
