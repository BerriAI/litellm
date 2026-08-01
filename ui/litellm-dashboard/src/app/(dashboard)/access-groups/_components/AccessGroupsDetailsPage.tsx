import { useAccessGroupDetails } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroupDetails";
import { ArrowLeftIcon, BotIcon, EditIcon, KeyIcon, LayersIcon, ServerIcon, UsersIcon } from "lucide-react";
import { useState } from "react";
import DefaultProxyAdminTag from "@/components/common_components/DefaultProxyAdminTag";
import CopyButton from "@/components/shared/CopyButton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { AccessGroupEditModal } from "./AccessGroupsModal/AccessGroupEditModal";

interface AccessGroupDetailProps {
  accessGroupId: string;
  onBack: () => void;
}

const MAX_PREVIEW = 5;

function ResourceList({ ids, emptyMessage }: { ids: string[]; emptyMessage: string }) {
  if (ids.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">{emptyMessage}</p>;
  }
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
      {ids.map((id) => (
        <Card key={id} size="sm">
          <CardContent>
            <code className="font-mono text-xs break-all text-foreground">{id}</code>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function AccessGroupDetail({ accessGroupId, onBack }: AccessGroupDetailProps) {
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
        <Button variant="ghost" size="icon" aria-label="Back" onClick={onBack} className="mb-4">
          <ArrowLeftIcon className="size-4" />
        </Button>
        <p className="py-8 text-center text-sm text-muted-foreground">Access group not found</p>
      </div>
    );
  }

  const modelIds = accessGroup.access_model_names ?? [];
  const mcpServerIds = accessGroup.access_mcp_server_ids ?? [];
  const agentIds = accessGroup.access_agent_ids ?? [];
  const keyIds = accessGroup.assigned_key_ids ?? [];
  const teamIds = accessGroup.assigned_team_ids ?? [];

  const displayedKeys = showAllKeys ? keyIds : keyIds.slice(0, MAX_PREVIEW);
  const displayedTeams = showAllTeams ? teamIds : teamIds.slice(0, MAX_PREVIEW);

  return (
    <div className="p-6 px-12">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" aria-label="Back" onClick={onBack}>
            <ArrowLeftIcon className="size-4" />
          </Button>
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-foreground">{accessGroup.access_group_name}</h1>
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <span>ID: {accessGroup.access_group_id}</span>
              <CopyButton value={accessGroup.access_group_id} label="Copy access group ID" />
            </div>
          </div>
        </div>
        <Button onClick={() => setIsEditModalVisible(true)}>
          <EditIcon className="size-4" />
          Edit Access Group
        </Button>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Group Details</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted-foreground">Description</dt>
            <dd className="text-foreground">{accessGroup.description || "—"}</dd>
            <dt className="text-muted-foreground">Created</dt>
            <dd className="flex items-center gap-1 text-foreground">
              {new Date(accessGroup.created_at).toLocaleString()}
              {accessGroup.created_by && (
                <>
                  <span>by</span>
                  <DefaultProxyAdminTag userId={accessGroup.created_by} />
                </>
              )}
            </dd>
            <dt className="text-muted-foreground">Last Updated</dt>
            <dd className="flex items-center gap-1 text-foreground">
              {new Date(accessGroup.updated_at).toLocaleString()}
              {accessGroup.updated_by && (
                <>
                  <span>by</span>
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
              Attached Keys
              <Badge variant="secondary">{keyIds.length}</Badge>
            </CardTitle>
            {keyIds.length > MAX_PREVIEW && (
              <CardAction>
                <Button variant="link" size="sm" onClick={() => setShowAllKeys(!showAllKeys)}>
                  {showAllKeys ? "Show Less" : `View All (${keyIds.length})`}
                </Button>
              </CardAction>
            )}
          </CardHeader>
          <CardContent>
            {keyIds.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {displayedKeys.map((id) => (
                  <Badge key={id} variant="secondary" className="font-mono">
                    {id.length > 20 ? `${id.slice(0, 10)}...${id.slice(-6)}` : id}
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No keys attached</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UsersIcon className="size-4" />
              Attached Teams
              <Badge variant="secondary">{teamIds.length}</Badge>
            </CardTitle>
            {teamIds.length > MAX_PREVIEW && (
              <CardAction>
                <Button variant="link" size="sm" onClick={() => setShowAllTeams(!showAllTeams)}>
                  {showAllTeams ? "Show Less" : `View All (${teamIds.length})`}
                </Button>
              </CardAction>
            )}
          </CardHeader>
          <CardContent>
            {teamIds.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {displayedTeams.map((id) => (
                  <Badge key={id} variant="secondary" className="font-mono">
                    {id}
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No teams attached</p>
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
                Models
                <Badge variant="secondary">{modelIds.length}</Badge>
              </TabsTrigger>
              <TabsTrigger value="mcp" className="flex-none gap-2 rounded-none px-4 py-2">
                <ServerIcon className="size-4" />
                MCP Servers
                <Badge variant="secondary">{mcpServerIds.length}</Badge>
              </TabsTrigger>
              <TabsTrigger value="agents" className="flex-none gap-2 rounded-none px-4 py-2">
                <BotIcon className="size-4" />
                Agents
                <Badge variant="secondary">{agentIds.length}</Badge>
              </TabsTrigger>
            </TabsList>
            <TabsContent value="models" className="pt-4">
              <ResourceList ids={modelIds} emptyMessage="No models assigned to this group" />
            </TabsContent>
            <TabsContent value="mcp" className="pt-4">
              <ResourceList ids={mcpServerIds} emptyMessage="No MCP servers assigned to this group" />
            </TabsContent>
            <TabsContent value="agents" className="pt-4">
              <ResourceList ids={agentIds} emptyMessage="No agents assigned to this group" />
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
