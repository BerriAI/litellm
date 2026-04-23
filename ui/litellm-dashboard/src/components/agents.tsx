import React, { useState, useEffect } from "react";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CheckCircle, Info } from "lucide-react";
import { getAgentsList, deleteAgentCall, keyListCall } from "./networking";
import AddAgentForm from "./agents/add_agent_form";
import { isAdminRole } from "@/utils/roles";
import AgentInfoView from "./agents/agent_info";
import NotificationsManager from "./molecules/notifications_manager";
import { Agent, AgentKeyInfo } from "./agents/types";
import { Team } from "./key_team_helpers/key_list";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import TableIconActionButton from "./common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";

interface AgentsPanelProps {
  accessToken: string | null;
  userRole?: string;
  teams?: Team[] | null;
}

interface AgentsResponse {
  agents: Agent[];
}

const AgentsPanel: React.FC<AgentsPanelProps> = ({
  accessToken,
  userRole,
  teams,
}) => {
  const [agentsList, setAgentsList] = useState<Agent[]>([]);
  const [keyInfoMap, setKeyInfoMap] = useState<Record<string, AgentKeyInfo>>(
    {},
  );
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [agentToDelete, setAgentToDelete] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [healthCheckEnabled, setHealthCheckEnabled] = useState(false);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchAgents = async (healthCheck?: boolean) => {
    if (!accessToken) return;
    setIsLoading(true);
    try {
      const response: AgentsResponse = await getAgentsList(
        accessToken,
        healthCheck ?? healthCheckEnabled,
      );
      setAgentsList(response.agents || []);
    } catch (error) {
      console.error("Error fetching agents:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchKeysForAgents = async () => {
    if (!accessToken) return;
    try {
      const { keys = [] } = await keyListCall(
        accessToken,
        null,
        null,
        null,
        null,
        null,
        1,
        500,
      );
      const map: Record<string, AgentKeyInfo> = {};
      for (const key of keys) {
        const agentId = (key as { agent_id?: string }).agent_id;
        if (agentId && !map[agentId]) {
          map[agentId] = {
            has_key: true,
            key_alias: (key as { key_alias?: string }).key_alias,
            token_prefix: (key as { token?: string }).token
              ? `${(key as { token: string }).token.slice(0, 8)}…`
              : undefined,
          };
        }
      }
      setKeyInfoMap(map);
    } catch (error) {
      console.error("Error fetching keys for agents:", error);
    }
  };

  useEffect(() => {
    fetchAgents();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  useEffect(() => {
    if (accessToken && agentsList.length > 0) {
      fetchKeysForAgents();
    } else if (agentsList.length === 0) {
      setKeyInfoMap({});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, agentsList.length]);

  const handleHealthCheckToggle = (checked: boolean) => {
    setHealthCheckEnabled(checked);
    fetchAgents(checked);
  };

  const handleAddAgent = () => {
    if (selectedAgentId) setSelectedAgentId(null);
    setIsAddModalVisible(true);
  };

  const handleDeleteConfirm = async () => {
    if (!agentToDelete || !accessToken) return;
    setIsDeleting(true);
    try {
      await deleteAgentCall(accessToken, agentToDelete.id);
      NotificationsManager.success(
        `Agent "${agentToDelete.name}" deleted successfully`,
      );
      fetchAgents();
    } catch (error) {
      console.error("Error deleting agent:", error);
      NotificationsManager.fromBackend("Failed to delete agent");
    } finally {
      setIsDeleting(false);
      setAgentToDelete(null);
    }
  };

  const sortedAgents = [...agentsList].sort((a, b) => {
    const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
    const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
    return dateB - dateA;
  });

  const columnCount = isAdmin ? 7 : 6;

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex flex-col gap-2 mb-4">
        <h1 className="text-2xl font-bold">Agents</h1>
        <p className="text-sm text-muted-foreground">
          List of A2A-spec agents that are available to be used in your
          organization. Go to AI Hub, to make agents public.
        </p>
        <Alert className="mb-3">
          <Info className="h-4 w-4" />
          <AlertTitle>Why do agents need keys?</AlertTitle>
          <AlertDescription>
            Keys scope access to an agent and allow it to call MCP tools.
            Assign a key when creating an agent or from the Virtual Keys page.
          </AlertDescription>
        </Alert>
        <div className="mt-2 flex items-center gap-4">
          {isAdmin && (
            <Button onClick={handleAddAgent} disabled={!accessToken}>
              + Add New Agent
            </Button>
          )}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-2">
                  <CheckCircle
                    className={
                      healthCheckEnabled
                        ? "h-4 w-4 text-emerald-500"
                        : "h-4 w-4 text-muted-foreground"
                    }
                  />
                  <span className="text-sm text-muted-foreground">
                    Health Check
                  </span>
                  <Switch
                    checked={healthCheckEnabled}
                    onCheckedChange={handleHealthCheckToggle}
                  />
                </div>
              </TooltipTrigger>
              <TooltipContent>
                When enabled, only agents with reachable URLs are shown
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      {selectedAgentId ? (
        <AgentInfoView
          agentId={selectedAgentId}
          onClose={() => setSelectedAgentId(null)}
          accessToken={accessToken}
          isAdmin={isAdmin}
        />
      ) : (
        <Card className="p-6">
          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-5 w-full" />
              <Skeleton className="h-5 w-full" />
              <Skeleton className="h-5 w-3/4" />
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Agent Name</TableHead>
                  <TableHead>Agent ID</TableHead>
                  <TableHead>Spend (USD)</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Status</TableHead>
                  {isAdmin && <TableHead>Actions</TableHead>}
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedAgents.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={columnCount}
                      className="text-center text-muted-foreground py-8"
                    >
                      No agents found. Click &quot;+ Add New Agent&quot; to
                      create one.
                    </TableCell>
                  </TableRow>
                ) : (
                  sortedAgents.map((agent) => (
                    <TableRow key={agent.agent_id}>
                      <TableCell>{agent.agent_name}</TableCell>
                      <TableCell>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="font-mono text-primary bg-primary/10 hover:bg-primary/20 text-xs font-normal px-2 py-0.5 h-auto text-left overflow-hidden truncate max-w-[200px]"
                                onClick={() =>
                                  setSelectedAgentId(agent.agent_id)
                                }
                              >
                                {agent.agent_id.slice(0, 7)}...
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>{agent.agent_id}</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </TableCell>
                      <TableCell>
                        {formatNumberWithCommas(agent.spend, 4)}
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">
                          {agent.litellm_params?.model || "N/A"}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {agent.created_at
                          ? new Date(agent.created_at).toLocaleDateString()
                          : "N/A"}
                      </TableCell>
                      <TableCell>
                        {keyInfoMap[agent.agent_id]?.has_key ? (
                          <Badge variant="default">Active</Badge>
                        ) : (
                          <Badge variant="secondary">Needs Setup</Badge>
                        )}
                      </TableCell>
                      {isAdmin && (
                        <TableCell>
                          <TableIconActionButton
                            variant="Delete"
                            onClick={() =>
                              setAgentToDelete({
                                id: agent.agent_id,
                                name: agent.agent_name,
                              })
                            }
                          />
                        </TableCell>
                      )}
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          )}
        </Card>
      )}

      <AddAgentForm
        visible={isAddModalVisible}
        onClose={() => setIsAddModalVisible(false)}
        accessToken={accessToken}
        onSuccess={fetchAgents}
        teams={teams}
      />

      <AlertDialog
        open={agentToDelete !== null}
        onOpenChange={(o) => (!o ? setAgentToDelete(null) : undefined)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Agent</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete agent: {agentToDelete?.name}?
              <br />
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteConfirm}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? "Deleting…" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default AgentsPanel;
