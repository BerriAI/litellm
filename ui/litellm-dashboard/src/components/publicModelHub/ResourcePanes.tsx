"use client";

import { Bot, Puzzle, Server } from "lucide-react";
import * as React from "react";
import { useTranslation } from "react-i18next";

import type { Plugin } from "@/components/claude_code_plugins/types";
import type { AgentCard, MCPServerData } from "@/components/PublicModelHubTableColumns";
import { ItemActions, ItemContent, ItemDescription, ItemMedia } from "@/components/ui/item";
import { matchesSearchTerm } from "@/utils/searchUtils";

import { AgentInspector } from "./AgentInspector";
import { FilterChip } from "./FilterChip";
import { mcpAuthLabel } from "./hubFormat";
import { HubMasterDetail } from "./HubMasterDetail";
import { LoadFailed, NoResults } from "./HubEmpty";
import { RowTitle, SearchInput } from "./HubParts";
import { McpInspector } from "./McpInspector";
import { SkillInspector } from "./SkillInspector";

interface ResourcePaneProps<T> {
  rows: T[];
  loading: boolean;
  error: boolean;
  onRetry: () => void;
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
  error,
  onRetry,
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
  const { t } = useTranslation();
  const [query, setQuery] = React.useState("");
  const [chosen, setChosen] = React.useState<string[]>([]);
  const options = [...new Set(rows.flatMap(filter.values))].sort();
  const sorted = [...rows].sort((a, b) => title(a).localeCompare(title(b)));
  const shown = sorted.filter(
    (row) =>
      matchesSearchTerm(query, search(row)) &&
      (chosen.length === 0 || filter.values(row).some((v) => chosen.includes(v))),
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
      opened={rows.find((row) => getKey(row) === selected)}
      getKey={getKey}
      onSelect={onSelect}
      empty={
        error ? (
          <LoadFailed noun={noun} onRetry={onRetry} />
        ) : (
          <NoResults icon={icon} noun={noun} query={query} onClear={filtered ? clear : undefined} />
        )
      }
      toolbar={
        <div className="flex items-center gap-2">
          <SearchInput
            value={query}
            onChange={setQuery}
            placeholder={t("publicModelHub.resourcePane.search", { noun })}
          />
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
  error: boolean;
  onRetry: () => void;
  selected: string | null;
  onSelect: (key: string | null) => void;
}

export function AgentsPane({ rows, ...props }: PaneProps<AgentCard>) {
  const { t } = useTranslation();
  return (
    <ResourcePane<AgentCard>
      {...props}
      rows={rows ?? []}
      noun={t("publicModelHub.agentsPane.noun")}
      icon={<Bot />}
      search={(agent) => [agent.name, agent.description]}
      filter={{
        label: t("publicModelHub.agentsPane.filterLabel"),
        noun: t("publicModelHub.agentsPane.filterNoun"),
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
  const { t } = useTranslation();
  return (
    <ResourcePane<MCPServerData>
      {...props}
      rows={rows ?? []}
      noun={t("publicModelHub.mcpPane.noun")}
      icon={<Server />}
      search={(server) => [server.name, server.mcp_info?.description]}
      filter={{
        label: t("publicModelHub.mcpPane.filterLabel"),
        noun: t("publicModelHub.mcpPane.filterNoun"),
        values: (server) => (server.transport ? [server.transport] : []),
      }}
      getKey={(server) => server.name}
      title={(server) => server.name}
      description={(server) => server.mcp_info?.description}
      meta={(server) => (server.auth_type ? mcpAuthLabel(server.auth_type) : undefined)}
      renderDetail={(server) => <McpInspector server={server} />}
    />
  );
}

export function SkillsPane({ rows, ...props }: PaneProps<Plugin>) {
  const { t } = useTranslation();
  return (
    <ResourcePane<Plugin>
      {...props}
      rows={rows ?? []}
      noun={t("publicModelHub.skillsPane.noun")}
      icon={<Puzzle />}
      search={(skill) => [skill.name, skill.description, skill.namespace, skill.domain, ...(skill.keywords ?? [])]}
      filter={{
        label: t("publicModelHub.skillsPane.filterLabel"),
        noun: t("publicModelHub.skillsPane.filterNoun"),
        values: (skill) => (skill.domain ? [skill.domain] : []),
      }}
      getKey={(skill) => skill.name}
      title={(skill) => skill.name}
      description={(skill) => skill.description}
      meta={(skill) => skill.domain}
      renderDetail={(skill) => <SkillInspector skill={skill} />}
    />
  );
}
