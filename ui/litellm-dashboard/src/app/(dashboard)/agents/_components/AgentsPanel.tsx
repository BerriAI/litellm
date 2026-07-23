import React, { useState, useEffect } from "react";
import { Modal, Alert } from "antd";
import { Plus } from "lucide-react";
import { getAgentsList, deleteAgentCall } from "@/components/networking";
import AddAgentForm from "./add_agent_form";
import { isAdminRole } from "@/utils/roles";
import AgentInfoView from "./agent_info";
import AgentsTable from "./AgentsTable";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { Agent } from "@/components/agents/types";
import { Team } from "@/components/key_team_helpers/key_list";
import { Button } from "@/components/ui/button";

interface AgentsPanelProps {
  accessToken: string | null;
  userRole?: string;
  teams?: Team[] | null;
}

interface AgentsResponse {
  agents: Agent[];
}

const AgentsPanel: React.FC<AgentsPanelProps> = ({ accessToken, userRole, teams }) => {
  const [agentsList, setAgentsList] = useState<Agent[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isHealthCheckLoading, setIsHealthCheckLoading] = useState(false);
  const [agentToDelete, setAgentToDelete] = useState<{ id: string; name: string } | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [healthCheckEnabled, setHealthCheckEnabled] = useState(false);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  useEffect(() => {
    let cancelled = false;
    const loadForToken = async () => {
      if (!accessToken) {
        setAgentsList([]);
        setIsLoading(false);
        return;
      }
      setIsLoading(true);
      try {
        const response: AgentsResponse = await getAgentsList(accessToken, false);
        if (!cancelled) {
          setAgentsList(response.agents || []);
        }
      } catch (error) {
        console.error("Error fetching agents:", error);
        if (!cancelled) {
          setAgentsList([]);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };
    loadForToken();
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  const refetchAgents = async (healthCheck: boolean) => {
    if (!accessToken) {
      return;
    }
    try {
      const response: AgentsResponse = await getAgentsList(accessToken, healthCheck);
      setAgentsList(response.agents || []);
    } catch (error) {
      console.error("Error fetching agents:", error);
    }
  };

  const handleHealthCheckToggle = async (checked: boolean) => {
    setHealthCheckEnabled(checked);
    setIsHealthCheckLoading(true);
    try {
      await refetchAgents(checked);
    } finally {
      setIsHealthCheckLoading(false);
    }
  };

  const handleAddAgent = () => {
    if (selectedAgentId) {
      setSelectedAgentId(null);
    }
    setIsAddModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleSuccess = () => {
    refetchAgents(healthCheckEnabled);
  };

  const handleDeleteClick = (agentId: string, agentName: string) => {
    setAgentToDelete({ id: agentId, name: agentName });
  };

  const handleDeleteConfirm = async () => {
    if (!agentToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deleteAgentCall(accessToken, agentToDelete.id);
      NotificationsManager.success(`Agent "${agentToDelete.name}" deleted successfully`);
      await refetchAgents(healthCheckEnabled);
    } catch (error) {
      console.error("Error deleting agent:", error);
      NotificationsManager.fromBackend("Failed to delete agent");
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
        <p className="text-sm text-gray-600">
          List of A2A-spec agents that are available to be used in your organization. Go to AI Hub, to make agents
          public.
        </p>
        <Alert
          message="Why do agents need keys?"
          description="Keys scope access to an agent and allow it to call MCP tools. Assign a key when creating an agent or from the Virtual Keys page."
          type="info"
          showIcon
          className="mb-3"
        />
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
        <AgentInfoView
          agentId={selectedAgentId}
          onClose={() => setSelectedAgentId(null)}
          accessToken={accessToken}
          isAdmin={isAdmin}
        />
      ) : (
        <AgentsTable
          agents={agentsList}
          isLoading={isLoading}
          isAdmin={isAdmin}
          healthCheckEnabled={healthCheckEnabled}
          isHealthCheckLoading={isHealthCheckLoading}
          onHealthCheckToggle={handleHealthCheckToggle}
          onAgentClick={(id) => setSelectedAgentId(id)}
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
        <Modal
          title="Delete Agent"
          open={agentToDelete !== null}
          onOk={handleDeleteConfirm}
          onCancel={handleDeleteCancel}
          confirmLoading={isDeleting}
          okText="Delete"
          okButtonProps={{ danger: true }}
        >
          <p>Are you sure you want to delete agent: {agentToDelete.name}?</p>
          <p>This action cannot be undone.</p>
        </Modal>
      )}
    </div>
  );
};

export default AgentsPanel;
