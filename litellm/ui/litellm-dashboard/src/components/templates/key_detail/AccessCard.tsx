"use client";

import { Bot, Database, Lock, Search, Server, Sparkles, type LucideIcon } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Input } from "@/components/ui/input";
import { BadgeLink } from "@/components/shared/BadgeLink";
import { mcpGrantsAllTools } from "@/components/mcp_server_management/effectiveMcpServers";
import { modelOrAccessGroupHref, teamDetailHref } from "@/utils/entityLinks";

import type { KeyAccess } from "./keyOverviewSections";
import { ResourceLink } from "./ResourceLink";
import { BadgeRow, Caption, CountPill, SectionCard, SubHead } from "./SectionCard";
import { useKeyAccessNames, type ResolvedMcpServer } from "./useKeyAccessNames";

const VISIBLE_MODELS = 8;

interface AccessCardProps {
  models: readonly string[];
  team: { id: string; name: string } | null;
  access: KeyAccess;
  accessGroupNames: ReadonlySet<string> | undefined;
}

export function AccessCard({ models, team, access, accessGroupNames }: AccessCardProps) {
  const { t } = useTranslation();
  const names = useKeyAccessNames(access);
  const cells = [
    access.mcp.kind !== "none" ? (
      <ToolCell key="mcp" icon={Server} title={t("keyDetail.overview.access.mcpServers")} count={mcpCount(access)}>
        <McpGrantView access={access} servers={names.mcpServers} toolsetName={names.toolsetName} />
      </ToolCell>
    ) : null,
    access.vectorStores.length > 0 ? (
      <ToolCell
        key="vector"
        icon={Database}
        title={t("keyDetail.overview.access.vectorStores")}
        count={access.vectorStores.length}
      >
        <BadgeRow>
          {access.vectorStores.map((id) => (
            <Badge key={id} variant="secondary" title={id}>
              {names.vectorStoreName(id)}
            </Badge>
          ))}
        </BadgeRow>
      </ToolCell>
    ) : null,
    access.agents.length > 0 || access.agentAccessGroups.length > 0 ? (
      <ToolCell
        key="agents"
        icon={Bot}
        title={t("keyDetail.overview.access.agents")}
        count={access.agents.length + access.agentAccessGroups.length}
      >
        <BadgeRow>
          {access.agents.map((id) => (
            <Badge key={id} variant="secondary" title={id}>
              {names.agentName(id)}
            </Badge>
          ))}
        </BadgeRow>
        {access.agentAccessGroups.length > 0 ? (
          <Caption>
            {t("keyDetail.overview.access.accessGroups", { groups: access.agentAccessGroups.join(", ") })}
          </Caption>
        ) : null}
      </ToolCell>
    ) : null,
    <ToolCell
      key="search"
      icon={Search}
      title={t("keyDetail.overview.access.searchTools")}
      count={access.searchTools.length || undefined}
    >
      {access.searchTools.length > 0 ? (
        <BadgeRow>
          {access.searchTools.map((tool) => (
            <Badge key={tool} variant="secondary">
              {tool}
            </Badge>
          ))}
        </BadgeRow>
      ) : (
        <div>{t("keyDetail.overview.access.allAllowed")}</div>
      )}
    </ToolCell>,
    access.skills.length > 0 ? (
      <ToolCell key="skills" icon={Sparkles} title={t("keyDetail.overview.access.skills")} count={access.skills.length}>
        <BadgeRow>
          {access.skills.map((skill) => (
            <Badge key={skill} variant="secondary">
              {skill}
            </Badge>
          ))}
        </BadgeRow>
        <Caption>{t("keyDetail.overview.access.skillsNote")}</Caption>
      </ToolCell>
    ) : null,
  ].filter((cell) => cell !== null);

  return (
    <SectionCard icon={Lock} title={t("keyDetail.overview.access.title")} testId="key-access-card">
      <ModelList models={models} team={team} accessGroupNames={accessGroupNames} />
      <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-x-6 gap-y-4 border-t border-border pt-3">
        {cells}
      </div>
    </SectionCard>
  );
}

const mcpCount = (access: KeyAccess): number | undefined =>
  access.mcp.kind === "list" ? access.mcp.servers.length + access.mcp.accessGroups.length : undefined;

function ToolCell({
  icon: Icon,
  title,
  count,
  children,
}: {
  icon: LucideIcon;
  title: string;
  count?: number;
  children: ReactNode;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <div className="flex items-center gap-1.5">
        <Icon className="size-4 text-muted-foreground" aria-hidden />
        <span className="font-medium">{title}</span>
        {count !== undefined ? <CountPill value={count} /> : null}
      </div>
      {children}
    </div>
  );
}

function McpGrantView({
  access,
  servers,
  toolsetName,
}: {
  access: KeyAccess;
  servers: readonly ResolvedMcpServer[];
  toolsetName: (id: string) => string;
}) {
  const { t } = useTranslation();
  if (access.mcp.kind === "blocked") return <div>{t("keyDetail.overview.access.mcpBlocked")}</div>;
  if (access.mcp.kind === "allProxy") return <div>{t("keyDetail.overview.access.mcpAllProxy")}</div>;
  if (access.mcp.kind !== "list") return null;
  const { accessGroups, toolsets } = access.mcp;
  return (
    <>
      {servers.length > 0 ? (
        <BadgeRow>
          {servers.map((server) => (
            <McpServerBadge key={server.identifier} server={server} />
          ))}
        </BadgeRow>
      ) : null}
      {toolsets.length > 0 ? (
        <Caption>{t("keyDetail.overview.access.toolsets", { toolsets: toolsets.map(toolsetName).join(", ") })}</Caption>
      ) : null}
      {accessGroups.length > 0 ? (
        <Caption>{t("keyDetail.overview.access.accessGroups", { groups: accessGroups.join(", ") })}</Caption>
      ) : null}
    </>
  );
}

function McpServerBadge({ server }: { server: ResolvedMcpServer }) {
  const { t } = useTranslation();
  const tools = server.allowedTools ?? [];
  if (tools.length === 0 || mcpGrantsAllTools(server.allowedTools)) {
    return (
      <Badge variant="secondary" title={server.identifier}>
        {server.name}
      </Badge>
    );
  }
  return (
    <HoverCard>
      <HoverCardTrigger
        render={
          <button type="button" className="inline-flex">
            <Badge variant="secondary">
              {t("keyDetail.overview.access.serverTools", { name: server.name, count: tools.length })}
            </Badge>
          </button>
        }
      />
      <HoverCardContent className="w-72">
        <div className="mb-1 text-xs font-medium text-muted-foreground">
          {t("keyDetail.overview.access.allowedTools", { name: server.name })}
        </div>
        <BadgeRow>
          {tools.map((tool) => (
            <Badge key={tool} variant="outline">
              {tool}
            </Badge>
          ))}
        </BadgeRow>
      </HoverCardContent>
    </HoverCard>
  );
}

function ModelList({
  models,
  team,
  accessGroupNames,
}: {
  models: readonly string[];
  team: { id: string; name: string } | null;
  accessGroupNames: ReadonlySet<string> | undefined;
}) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const matching = needle ? models.filter((model) => model.toLowerCase().includes(needle)) : models;
  const shown = expanded ? matching : matching.slice(0, VISIBLE_MODELS);

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <SubHead title={t("keyDetail.overview.access.models")} count={models.length || undefined} />
      {models.length === 0 ? (
        <p className="m-0">
          {team ? (
            <>
              {t("keyDetail.overview.access.everyTeamModelPrefix")}
              <ResourceLink href={teamDetailHref(team.id)}>{team.name}</ResourceLink>
              {t("keyDetail.overview.access.everyTeamModelSuffix")}
            </>
          ) : (
            t("keyDetail.overview.access.everyProxyModel")
          )}
          <span className="text-muted-foreground">{t("keyDetail.overview.access.noModelList")}</span>
        </p>
      ) : null}
      {expanded && models.length > VISIBLE_MODELS ? (
        <Input
          className="h-8 max-w-[280px]"
          placeholder={t("keyDetail.overview.access.searchModels")}
          aria-label={t("keyDetail.overview.access.searchModels")}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      ) : null}
      {models.length > 0 ? (
        <BadgeRow>
          {shown.map((model) => (
            <BadgeLink
              key={model}
              href={modelOrAccessGroupHref(model, accessGroupNames)}
              className="min-w-0 break-words"
            >
              {model}
            </BadgeLink>
          ))}
          {!expanded && models.length > VISIBLE_MODELS ? (
            <Button variant="ghost" size="sm" onClick={() => setExpanded(true)}>
              {t("keyDetail.overview.access.moreModels", { count: models.length - VISIBLE_MODELS })}
            </Button>
          ) : null}
        </BadgeRow>
      ) : null}
      {expanded ? (
        <div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setExpanded(false);
              setQuery("");
            }}
          >
            {t("keyDetail.overview.limitTable.showFewer")}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
