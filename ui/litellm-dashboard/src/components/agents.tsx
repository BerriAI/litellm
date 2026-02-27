import React, { useState, useEffect } from "react";
import { Button } from "@tremor/react";
import { Modal, Alert } from "antd";
import { getAgentsList, deleteAgentCall, keyListCall } from "./networking";
import AddAgentForm from "./agents/add_agent_form";
import AgentCardGrid from "./agents/agent_card_grid";
import { isAdminRole } from "@/utils/roles";
import AgentInfoView from "./agents/agent_info";
import NotificationsManager from "./molecules/notifications_manager";
import { Agent, AgentKeyInfo } from "./agents/types";

interface AgentsPanelProps {
  accessToken: string | null;
  userRole?: string;
}

interface AgentsResponse {
  agents: Agent[];
}

const AgentsPanel: React.FC<AgentsPanelProps> = ({ accessToken, userRole }) => {
  const [agentsList, setAgentsList] = useState<Agent[]>([]);
  const [keyInfoMap, setKeyInfoMap] = useState<Record<string, AgentKeyInfo>>({});
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
        500
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
  }, [accessToken]);

  useEffect(() => {
    if (accessToken && agentsList.length > 0) {
      fetchKeysForAgents();
    } else if (agentsList.length === 0) {
      setKeyInfoMap({});
    }
  }, [accessToken, agentsList.length]);

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
      <div className="flex flex-col gap-2 mb-4">
        <h1 className="text-2xl font-bold">Agents</h1>
        <p className="text-sm text-gray-600">List of A2A-spec agents that are available to be used in your organization. Go to AI Hub, to make agents public.</p>
        <Alert
          message="Why do agents need keys?"
          description="Keys scope access to an agent and allow it to call MCP tools. Assign a key when creating an agent or from the Virtual Keys page."
          type="info"
          showIcon
          className="mb-3"
        />
        <div className="mt-2">
          <Button onClick={handleAddAgent} disabled={!accessToken}>
            + Add New Agent
          </Button>
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
        <AgentCardGrid
          agentsList={agentsList}
          keyInfoMap={keyInfoMap}
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

