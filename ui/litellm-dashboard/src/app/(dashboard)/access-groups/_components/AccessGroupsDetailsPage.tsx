import { useAccessGroupDetails } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroupDetails";
import type { TFunction } from "i18next";
import {
  ArrowLeftIcon,
  BotIcon,
  EditIcon,
  KeyIcon,
  LayersIcon,
  ServerIcon,
  SparklesIcon,
  UsersIcon,
} from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import DefaultProxyAdminTag from "@/components/common_components/DefaultProxyAdminTag";
import { BadgeLink } from "@/components/shared/BadgeLink";
import CopyButton from "@/components/shared/CopyButton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import type { components } from "@/lib/http/schema";
import { keyDetailHref, teamDetailHref } from "@/utils/entityLinks";
import { AccessGroupEditModal } from "./AccessGroupsModal/AccessGroupEditModal";

type AccessGroupResource = components["schemas"]["AccessGroupResource"];

interface AccessGroupDetailProps {
  accessGroupId: string;
  onBack: () => void;
}

const MAX_PREVIEW = 5;

const shortId = (id: string) => (id.length > 20 ? `${id.slice(0, 10)}...${id.slice(-6)}` : id);

const describeAllowedTools = (tools: readonly string[] | undefined, t: TFunction): string => {
  if (tools === undefined) {
    return t("accessGroups.detail.toolPermissions.allTools");
  }
  return tools.length === 0 ? t("accessGroups.detail.toolPermissions.noTools") : tools.join(", ");
};

function MCPToolPermissionList({
  servers,
  toolPermissions,
}: {
  servers: readonly AccessGroupResource[];
  toolPermissions: Readonly<Record<string, string[]>> | null;
}) {
  const { t } = useTranslation();
  if (toolPermissions === null || servers.length === 0) {
    return null;
  }
  return (
    <section className="mt-6 space-y-2" aria-label={t("accessGroups.detail.toolPermissions.label")}>
      {servers.map(({ id, name }) => {
        return (
          <div key={id} className="flex flex-wrap items-baseline gap-2 text-sm">
            <span className="font-medium">{name ?? id}</span>
            <span className="text-muted-foreground">{describeAllowedTools(toolPermissions[id], t)}</span>
          </div>
        );
      })}
    </section>
  );
}

function ResourceList({ items, emptyMessage }: { items: readonly AccessGroupResource[]; emptyMessage: string }) {
  if (items.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">{emptyMessage}</p>;
  }
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
      {items.map(({ id, name }) => (
        <Card key={id} size="sm">
          <CardContent>
            {name ? (
              <SimpleTooltip content={id}>
                <span className="text-sm font-medium break-all text-foreground">{name}</span>
              </SimpleTooltip>
            ) : (
              <code className="font-mono text-xs break-all text-foreground">{id}</code>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function ResourceBadge({
  resource: { id, name },
  href,
  fallback,
}: {
  resource: AccessGroupResource;
  href?: string;
  fallback: (id: string) => string;
}) {
  const badge = (
    <BadgeLink href={href} className={name ? undefined : "font-mono"}>
      {name ?? fallback(id)}
    </BadgeLink>
  );
  return name ? <SimpleTooltip content={id}>{badge}</SimpleTooltip> : badge;
}

export function AccessGroupDetail({ accessGroupId, onBack }: AccessGroupDetailProps) {
  const { t } = useTranslation();
  const { data: accessGroup, isLoading } = useAccessGroupDetails(accessGroupId);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [showAllKeys, setShowAllKeys] = useState(false);
  const [showAllTeams, setShowAllTeams] = useState(false);

  if (isLoading) {
    return (
      <div className="p-6 px-12">
        <div className="flex min-h-[300px] items-center justify-center">
          <UiLoadingSpinner className="size-8 text-primary" />
        </div>
      </div>
    );
  }

  if (!accessGroup) {
    return (
      <div className="p-6 px-12">
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("accessGroups.detail.back")}
          onClick={onBack}
          className="mb-4"
        >
          <ArrowLeftIcon className="size-4" />
        </Button>
        <p className="py-8 text-center text-sm text-muted-foreground">{t("accessGroups.detail.notFound")}</p>
      </div>
    );
  }

  const models = accessGroup.access_model_names.map((id) => ({ id, name: null }));
  const mcpServers = accessGroup.access_mcp_servers;
  const agents = accessGroup.access_agents;
  const skills = (accessGroup.access_skill_names ?? []).map((name) => ({ id: name, name: null }));
  const keys = accessGroup.assigned_keys;
  const teams = accessGroup.assigned_teams;

  const displayedKeys = showAllKeys ? keys : keys.slice(0, MAX_PREVIEW);
  const displayedTeams = showAllTeams ? teams : teams.slice(0, MAX_PREVIEW);

  return (
    <div className="p-6 px-12">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" aria-label={t("accessGroups.detail.back")} onClick={onBack}>
            <ArrowLeftIcon className="size-4" />
          </Button>
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-foreground">{accessGroup.access_group_name}</h1>
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <span>
                {t("accessGroups.detail.idLabel")}
                {accessGroup.access_group_id}
              </span>
              <CopyButton value={accessGroup.access_group_id} label={t("accessGroups.detail.copyId")} />
            </div>
          </div>
        </div>
        <Button onClick={() => setIsEditModalVisible(true)}>
          <EditIcon className="size-4" />
          {t("accessGroups.detail.edit")}
        </Button>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>{t("accessGroups.detail.groupDetails")}</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted-foreground">{t("accessGroups.detail.description")}</dt>
            <dd className="text-foreground">{accessGroup.description || "—"}</dd>
            <dt className="text-muted-foreground">{t("accessGroups.detail.created")}</dt>
            <dd className="flex items-center gap-1 text-foreground">
              {new Date(accessGroup.created_at).toLocaleString()}
              {accessGroup.created_by && (
                <>
                  <span>{t("accessGroups.detail.by")}</span>
                  <DefaultProxyAdminTag userId={accessGroup.created_by} />
                </>
              )}
            </dd>
            <dt className="text-muted-foreground">{t("accessGroups.detail.lastUpdated")}</dt>
            <dd className="flex items-center gap-1 text-foreground">
              {new Date(accessGroup.updated_at).toLocaleString()}
              {accessGroup.updated_by && (
                <>
                  <span>{t("accessGroups.detail.by")}</span>
                  <DefaultProxyAdminTag userId={accessGroup.updated_by} />
                </>
              )}
            </dd>
          </dl>
        </CardContent>
      </Card>

      <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <KeyIcon className="size-4" />
              {t("accessGroups.detail.attachedKeys")}
              <Badge variant="secondary">{keys.length}</Badge>
            </CardTitle>
            {keys.length > MAX_PREVIEW && (
              <CardAction>
                <Button variant="link" size="sm" onClick={() => setShowAllKeys(!showAllKeys)}>
                  {showAllKeys
                    ? t("accessGroups.detail.showLess")
                    : t("accessGroups.detail.viewAll", { count: keys.length })}
                </Button>
              </CardAction>
            )}
          </CardHeader>
          <CardContent>
            {keys.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {displayedKeys.map((key) => (
                  <ResourceBadge key={key.id} resource={key} href={keyDetailHref(key.id)} fallback={shortId} />
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">{t("accessGroups.detail.noKeys")}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UsersIcon className="size-4" />
              {t("accessGroups.detail.attachedTeams")}
              <Badge variant="secondary">{teams.length}</Badge>
            </CardTitle>
            {teams.length > MAX_PREVIEW && (
              <CardAction>
                <Button variant="link" size="sm" onClick={() => setShowAllTeams(!showAllTeams)}>
                  {showAllTeams
                    ? t("accessGroups.detail.showLess")
                    : t("accessGroups.detail.viewAll", { count: teams.length })}
                </Button>
              </CardAction>
            )}
          </CardHeader>
          <CardContent>
            {teams.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {displayedTeams.map((team) => (
                  <ResourceBadge key={team.id} resource={team} href={teamDetailHref(team.id)} fallback={(id) => id} />
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">{t("accessGroups.detail.noTeams")}</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardContent>
          <Tabs defaultValue="models">
            <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0">
              <TabsTrigger value="models" className="flex-none gap-2 rounded-none px-4 py-2">
                <LayersIcon className="size-4" />
                {t("accessGroups.detail.tabs.models")}
                <Badge variant="secondary">{models.length}</Badge>
              </TabsTrigger>
              <TabsTrigger value="mcp" className="flex-none gap-2 rounded-none px-4 py-2">
                <ServerIcon className="size-4" />
                {t("accessGroups.detail.tabs.mcpServers")}
                <Badge variant="secondary">{mcpServers.length}</Badge>
              </TabsTrigger>
              <TabsTrigger value="agents" className="flex-none gap-2 rounded-none px-4 py-2">
                <BotIcon className="size-4" />
                {t("accessGroups.detail.tabs.agents")}
                <Badge variant="secondary">{agents.length}</Badge>
              </TabsTrigger>
              <TabsTrigger value="skills" className="flex-none gap-2 rounded-none px-4 py-2">
                <SparklesIcon className="size-4" />
                {t("accessGroups.detail.tabs.skills")}
                <Badge variant="secondary">{skills.length}</Badge>
              </TabsTrigger>
            </TabsList>
            <TabsContent value="models" className="pt-4">
              <ResourceList items={models} emptyMessage={t("accessGroups.detail.empty.models")} />
            </TabsContent>
            <TabsContent value="mcp" className="pt-4">
              <ResourceList items={mcpServers} emptyMessage={t("accessGroups.detail.empty.mcpServers")} />
              <MCPToolPermissionList servers={mcpServers} toolPermissions={accessGroup.mcp_tool_permissions ?? null} />
            </TabsContent>
            <TabsContent value="agents" className="pt-4">
              <ResourceList items={agents} emptyMessage={t("accessGroups.detail.empty.agents")} />
            </TabsContent>
            <TabsContent value="skills" className="pt-4">
              <ResourceList items={skills} emptyMessage={t("accessGroups.detail.empty.skills")} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <AccessGroupEditModal
        visible={isEditModalVisible}
        accessGroup={accessGroup}
        onCancel={() => setIsEditModalVisible(false)}
      />
    </div>
  );
}
