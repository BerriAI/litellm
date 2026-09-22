import React, { useEffect, useReducer, useState } from "react";
import { Info, Plus } from "lucide-react";
import { getAgentsList, deleteAgentCall } from "@/components/networking";
import AddAgentForm from "./add_agent_form";
import { isAdminRole } from "@/utils/roles";
import AgentInfoView from "./agent_info";
import AgentsTable from "./AgentsTable";
import { toast } from "@/lib/toast";
import { Agent } from "@/components/agents/types";
import { Team } from "@/components/key_team_helpers/key_list";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { useAgentDetailUrlState, useAgentHealthCheck } from "./useAgentUrlState";

interface AgentsPanelProps {
  accessToken: string | null;
  userRole?: string;
  teams?: Team[] | null;
}

interface AgentsResponse {
  agents: Agent[];
}

interface AgentsLoad {
  accessToken: string;
  healthCheck: boolean;
  agents: Agent[];
}

const NO_AGENTS: Agent[] = [];

const fetchAgents = async (accessToken: string, healthCheck: boolean): Promise<Agent[] | null> => {
  try {
    const response: AgentsResponse = await getAgentsList(accessToken, healthCheck);
    return response.agents || [];
  } catch (error) {
    console.error("Error fetching agents:", error);
    return null;
  }
};

const settleAgentsLoad = (
  previous: AgentsLoad | null,
  request: Omit<AgentsLoad, "agents">,
  agents: Agent[] | null,
): AgentsLoad => ({
  ...request,
  agents: agents ?? (previous?.accessToken === request.accessToken ? previous.agents : NO_AGENTS),
});

const AgentsPanel: React.FC<AgentsPanelProps> = ({ accessToken, userRole, teams }) => {
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [agentToDelete, setAgentToDelete] = useState<{ id: string; name: string } | null>(null);
  const { selectedAgentId, openAgent, closeAgent } = useAgentDetailUrlState();
  const [healthCheckEnabled, setHealthCheckEnabled] = useAgentHealthCheck();
  const [agentsLoad, setAgentsLoad] = useState<AgentsLoad | null>(null);
  const [reloadCount, reloadAgents] = useReducer((count: number) => count + 1, 0);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    const request = { accessToken, healthCheck: healthCheckEnabled };
    void fetchAgents(accessToken, healthCheckEnabled).then((agents) => {
      if (!cancelled) setAgentsLoad((previous) => settleAgentsLoad(previous, request, agents));
    });
    return () => {
      cancelled = true;
    };
  }, [accessToken, healthCheckEnabled, reloadCount]);

  const currentLoad = accessToken && agentsLoad?.accessToken === accessToken ? agentsLoad : null;
  const agentsList = currentLoad?.agents ?? NO_AGENTS;
  const isLoading = Boolean(accessToken) && currentLoad === null;
  const isHealthCheckLoading = currentLoad !== null && currentLoad.healthCheck !== healthCheckEnabled;

  const handleHealthCheckToggle = (checked: boolean) => {
    void setHealthCheckEnabled(checked);
  };

  const handleAddAgent = () => {
    if (selectedAgentId) {
      closeAgent();
    }
    setIsAddModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleSuccess = () => {
    reloadAgents();
  };

  const handleDeleteClick = (agentId: string, agentName: string) => {
    setAgentToDelete({ id: agentId, name: agentName });
  };

  const handleDeleteConfirm = async () => {
    if (!agentToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deleteAgentCall(accessToken, agentToDelete.id);
      toast.success(`Agent "${agentToDelete.name}" deleted successfully`);
      const request = { accessToken, healthCheck: healthCheckEnabled };
      const agents = await fetchAgents(accessToken, healthCheckEnabled);
      setAgentsLoad((previous) => settleAgentsLoad(previous, request, agents));
    } catch (error) {
      console.error("Error deleting agent:", error);
      toast.fromError("Failed to delete agent");
    } finally {
      setIsDeleting(false);
      setAgentToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setAgentToDelete(null);
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex flex-col gap-2 mb-4">
        <h1 className="text-2xl font-bold">Agents</h1>
        <p className="text-sm text-muted-foreground">
          List of A2A-spec agents that are available to be used in your organization. Go to AI Hub, to make agents
          public.
        </p>
        <Alert className="mb-3">
          <Info />
          <AlertTitle>Why do agents need keys?</AlertTitle>
          <AlertDescription>
            Keys scope access to an agent and allow it to call MCP tools. Assign a key when creating an agent or from
            the Virtual Keys page.
          </AlertDescription>
        </Alert>
        {isAdmin && (
          <div className="mt-2 flex items-center gap-4">
            <Button onClick={handleAddAgent} disabled={!accessToken}>
              <Plus />
              Add New Agent
            </Button>
          </div>
        )}
      </div>

      {selectedAgentId ? (
        <AgentInfoView agentId={selectedAgentId} onClose={closeAgent} accessToken={accessToken} isAdmin={isAdmin} />
      ) : (
        <AgentsTable
          agents={agentsList}
          isLoading={isLoading}
          isAdmin={isAdmin}
          healthCheckEnabled={healthCheckEnabled}
          isHealthCheckLoading={isHealthCheckLoading}
          onHealthCheckToggle={handleHealthCheckToggle}
          onAgentClick={openAgent}
          onDeleteClick={handleDeleteClick}
        />
      )}

      <AddAgentForm
        visible={isAddModalVisible}
        onClose={handleCloseModal}
        accessToken={accessToken}
        onSuccess={handleSuccess}
        teams={teams}
      />

      {agentToDelete && (
        <AlertDialog
          open
          onOpenChange={(open) => {
            if (!open) handleDeleteCancel();
          }}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete Agent</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete agent: {agentToDelete.name}? This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <Button variant="destructive" onClick={handleDeleteConfirm} disabled={isDeleting}>
                Delete
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
};

export default AgentsPanel;
