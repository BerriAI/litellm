import React, { useState, useEffect } from "react";
import { Button } from "@tremor/react";
import { Modal } from "antd";
import { getAgentsList, deleteAgentCall } from "./networking";
import AddAgentForm from "./agents/add_agent_form";
import AgentTable from "./agents/agent_table";
import { isAdminRole } from "@/utils/roles";
import AgentInfoView from "./agents/agent_info";
import NotificationsManager from "./molecules/notifications_manager";
import { Agent } from "./agents/types";

interface AgentsPanelProps {
  accessToken: string | null;
  userRole?: string;
}

interface AgentsResponse {
  agents: Agent[];
}

const AgentsPanel: React.FC<AgentsPanelProps> = ({ accessToken, userRole }) => {
  const [agentsList, setAgentsList] = useState<Agent[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [agentToDelete, setAgentToDelete] = useState<{ id: string; name: string } | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchAgents = async () => {
    if (!accessToken) {
      return;
    }

    setIsLoading(true);
    try {
      const response: AgentsResponse = await getAgentsList(accessToken);
      console.log(`agents: ${JSON.stringify(response)}`);
      setAgentsList(response.agents);
    } catch (error) {
      console.error("Error fetching agents:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAgents();
  }, [accessToken]);

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
    fetchAgents();
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
      fetchAgents();
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
      <div className="flex justify-between items-center mb-4">
        <div className="flex-col gap-2">
          <h1 className="text-2xl font-bold">Agents</h1>
          <p className="text-sm text-gray-600">List of A2A-spec agents that are available to be used in your organization. Go to AI Hub, to make agents public.</p>
        </div>
        <Button onClick={handleAddAgent} disabled={!accessToken}>
          + Add New Agent
        </Button>
      </div>

      {selectedAgentId ? (
        <AgentInfoView
          agentId={selectedAgentId}
          onClose={() => setSelectedAgentId(null)}
          accessToken={accessToken}
          isAdmin={isAdmin}
        />
      ) : (
        <AgentTable
          agentsList={agentsList}
          isLoading={isLoading}
          onDeleteClick={handleDeleteClick}
          accessToken={accessToken}
          onAgentUpdated={fetchAgents}
          isAdmin={isAdmin}
          onAgentClick={(id) => setSelectedAgentId(id)}
        />
      )}

      <AddAgentForm
        visible={isAddModalVisible}
        onClose={handleCloseModal}
        accessToken={accessToken}
        onSuccess={handleSuccess}
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

