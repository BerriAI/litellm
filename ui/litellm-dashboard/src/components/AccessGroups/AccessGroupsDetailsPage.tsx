import { useAccessGroupDetails } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroupDetails";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";
import {
  ArrowLeftIcon,
  BotIcon,
  CopyIcon,
  EditIcon,
  KeyIcon,
  LayersIcon,
  ServerIcon,
  UsersIcon,
} from "lucide-react";
import { useState } from "react";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import { AccessGroupEditModal } from "./AccessGroupsModal/AccessGroupEditModal";

interface AccessGroupDetailProps {
  accessGroupId: string;
  onBack: () => void;
}

function EmptyState({ description }: { description: string }) {
  return (
    <div className="py-12 flex flex-col items-center justify-center text-muted-foreground">
      <div className="text-sm">{description}</div>
    </div>
  );
}

function CopyableId({ value }: { value: string }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => {
              navigator.clipboard?.writeText(value);
              toast.success("Copied to clipboard");
            }}
            className="inline-flex items-center gap-1 text-xs font-mono rounded bg-muted px-1.5 py-0.5 hover:bg-muted/80"
          >
            {value}
            <CopyIcon size={12} />
          </button>
        </TooltipTrigger>
        <TooltipContent>Copy</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function AccessGroupDetail({
  accessGroupId,
  onBack,
}: AccessGroupDetailProps) {
  const { data: accessGroup, isLoading } =
    useAccessGroupDetails(accessGroupId);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [showAllKeys, setShowAllKeys] = useState(false);
  const [showAllTeams, setShowAllTeams] = useState(false);

  const MAX_PREVIEW = 5;

  if (isLoading) {
    return (
      <div className="p-6 md:px-12">
        <div className="flex justify-center items-center min-h-[300px]">
          <Skeleton className="h-10 w-10 rounded-full" />
        </div>
      </div>
    );
  }

  if (!accessGroup) {
    return (
      <div className="p-6 md:px-12">
        <Button variant="ghost" size="sm" onClick={onBack} className="mb-4">
          <ArrowLeftIcon size={16} />
        </Button>
        <EmptyState description="Access group not found" />
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

  const handleEdit = () => setIsEditModalVisible(true);

  return (
    <div className="p-6 md:px-12">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={onBack}>
            <ArrowLeftIcon size={16} />
          </Button>
          <div>
            <h2 className="text-2xl font-semibold m-0">
              {accessGroup.access_group_name}
            </h2>
            <div className="text-sm text-muted-foreground flex items-center gap-2">
              ID: <CopyableId value={accessGroup.access_group_id} />
            </div>
          </div>
        </div>
        <Button onClick={handleEdit}>
          <EditIcon size={16} />
          Edit Access Group
        </Button>
      </div>

      {/* Group Details */}
      <Card className="p-6 mb-6">
        <h3 className="text-lg font-semibold mb-3">Group Details</h3>
        <dl className="space-y-2 text-sm">
          <div className="flex gap-2">
            <dt className="w-36 text-muted-foreground">Description</dt>
            <dd>{accessGroup.description || "—"}</dd>
          </div>
          <div className="flex gap-2">
            <dt className="w-36 text-muted-foreground">Created</dt>
            <dd>
              {new Date(accessGroup.created_at).toLocaleString()}
              {accessGroup.created_by && (
                <>
                  &nbsp;by&nbsp;
                  <DefaultProxyAdminTag userId={accessGroup.created_by} />
                </>
              )}
            </dd>
          </div>
          <div className="flex gap-2">
            <dt className="w-36 text-muted-foreground">Last Updated</dt>
            <dd>
              {new Date(accessGroup.updated_at).toLocaleString()}
              {accessGroup.updated_by && (
                <>
                  &nbsp;by&nbsp;
                  <DefaultProxyAdminTag userId={accessGroup.updated_by} />
                </>
              )}
            </dd>
          </div>
        </dl>
      </Card>

      {/* Attached Keys & Teams */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-2 font-semibold">
              <KeyIcon size={16} />
              Attached Keys
              <Badge variant="secondary">{keyIds?.length}</Badge>
            </div>
            {keyIds?.length > MAX_PREVIEW && (
              <Button
                variant="link"
                size="sm"
                onClick={() => setShowAllKeys(!showAllKeys)}
              >
                {showAllKeys ? "Show Less" : `View All (${keyIds?.length})`}
              </Button>
            )}
          </div>
          {keyIds?.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {displayedKeys.map((id) => (
                <Badge key={id} variant="secondary" className="font-mono text-xs">
                  {id.length > 20
                    ? `${id.slice(0, 10)}...${id.slice(-6)}`
                    : id}
                </Badge>
              ))}
            </div>
          ) : (
            <EmptyState description="No keys attached" />
          )}
        </Card>
        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-2 font-semibold">
              <UsersIcon size={16} />
              Attached Teams
              <Badge variant="secondary">{teamIds?.length}</Badge>
            </div>
            {teamIds?.length > MAX_PREVIEW && (
              <Button
                variant="link"
                size="sm"
                onClick={() => setShowAllTeams(!showAllTeams)}
              >
                {showAllTeams ? "Show Less" : `View All (${teamIds?.length})`}
              </Button>
            )}
          </div>
          {teamIds?.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {displayedTeams.map((id) => (
                <Badge key={id} variant="secondary" className="font-mono text-xs">
                  {id}
                </Badge>
              ))}
            </div>
          ) : (
            <EmptyState description="No teams attached" />
          )}
        </Card>
      </div>

      {/* Resources Tabs */}
      <Card className="p-6">
        <Tabs defaultValue="models" className="w-full">
          <TabsList>
            <TabsTrigger value="models" className="gap-1">
              <LayersIcon size={16} />
              Models
              <Badge variant="secondary">{modelIds?.length}</Badge>
            </TabsTrigger>
            <TabsTrigger value="mcp" className="gap-1">
              <ServerIcon size={16} />
              MCP Servers
              <Badge variant="secondary">{mcpServerIds?.length}</Badge>
            </TabsTrigger>
            <TabsTrigger value="agents" className="gap-1">
              <BotIcon size={16} />
              Agents
              <Badge variant="secondary">{agentIds?.length}</Badge>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="models" className="pt-4">
            {modelIds?.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {modelIds.map((id) => (
                  <Card key={id} className="p-3">
                    <code className="text-xs font-mono">{id}</code>
                  </Card>
                ))}
              </div>
            ) : (
              <EmptyState description="No models assigned to this group" />
            )}
          </TabsContent>

          <TabsContent value="mcp" className="pt-4">
            {mcpServerIds?.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {mcpServerIds.map((id) => (
                  <Card key={id} className="p-3">
                    <code className="text-xs font-mono">{id}</code>
                  </Card>
                ))}
              </div>
            ) : (
              <EmptyState description="No MCP servers assigned to this group" />
            )}
          </TabsContent>

          <TabsContent value="agents" className="pt-4">
            {agentIds?.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {agentIds.map((id) => (
                  <Card key={id} className="p-3">
                    <code className="text-xs font-mono">{id}</code>
                  </Card>
                ))}
              </div>
            ) : (
              <EmptyState description="No agents assigned to this group" />
            )}
          </TabsContent>
        </Tabs>
      </Card>

      {/* Edit Modal */}
      <AccessGroupEditModal
        visible={isEditModalVisible}
        accessGroup={accessGroup}
        onCancel={() => setIsEditModalVisible(false)}
      />
    </div>
  );
}
