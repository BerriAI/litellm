import React, { useState, useEffect } from "react";
import {
  Button,
  Card,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Badge,
  Text,
} from "@tremor/react";
import { Modal, Alert, Tooltip, Skeleton, Switch } from "antd";
import { CheckCircleOutlined } from "@ant-design/icons";
import { getAgentsList, deleteAgentCall, keyListCall } from "./networking";
import AddAgentForm from "./agents/add_agent_form";
import { isAdminRole } from "@/utils/roles";
import AgentInfoView from "./agents/agent_info";
import NotificationsManager from "./molecules/notifications_manager";
import { Agent, AgentKeyInfo } from "./agents/types";
import { Team } from "./key_team_helpers/key_list";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import TableIconActionButton from "./common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { getMockDriftSummary, DriftBadgeCell } from "./agents/agent_drift_helpers";

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
  const [keyInfoMap, setKeyInfoMap] = useState<Record<string, AgentKeyInfo>>({});
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [agentToDelete, setAgentToDelete] = useState<{ id: string; name: string } | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [healthCheckEnabled, setHealthCheckEnabled] = useState(false);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchAgents = async (healthCheck?: boolean) => {
    if (!accessToken) {
      return;
    }

    setIsLoading(true);
    try {
      const response: AgentsResponse = await getAgentsList(accessToken, healthCheck ?? healthCheckEnabled);
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

  const handleHealthCheckToggle = (checked: boolean) => {
    setHealthCheckEnabled(checked);
    fetchAgents(checked);
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

  const sortedAgents = [...agentsList].sort((a, b) => {
    const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
    const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
    return dateB - dateA;
  });

  const columnCount = isAdmin ? 8 : 7;

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
        <div className="mt-2 flex items-center gap-4">
          {isAdmin && (
            <Button onClick={handleAddAgent} disabled={!accessToken}>
              + Add New Agent
            </Button>
          )}
          <Tooltip title="When enabled, only agents with reachable URLs are shown">
            <div className="flex items-center gap-2">
              <CheckCircleOutlined className={healthCheckEnabled ? "text-green-500" : "text-gray-400"} />
              <span className="text-sm text-gray-600">Health Check</span>
              <Switch
                size="small"
                checked={healthCheckEnabled}
                onChange={handleHealthCheckToggle}
                loading={isLoading && healthCheckEnabled}
              />
            </div>
          </Tooltip>
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
        <Card>
          {isLoading ? (
            <Skeleton active paragraph={{ rows: 3 }} />
          ) : (
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell>Agent Name</TableHeaderCell>
                  <TableHeaderCell>Agent ID</TableHeaderCell>
                  <TableHeaderCell>Spend (USD)</TableHeaderCell>
                  <TableHeaderCell>Model</TableHeaderCell>
                  <TableHeaderCell>Created</TableHeaderCell>
                  <TableHeaderCell>Status</TableHeaderCell>
                  <TableHeaderCell>Drift</TableHeaderCell>
                  {isAdmin && <TableHeaderCell>Actions</TableHeaderCell>}
                </TableRow>
              </TableHead>
              <TableBody>
                {sortedAgents.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={columnCount}>
                      <Text className="text-center">No agents found. Click &quot;+ Add New Agent&quot; to create one.</Text>
                    </TableCell>
                  </TableRow>
                ) : (
                  sortedAgents.map((agent) => {
                    const drift = getMockDriftSummary(agent.agent_id);
                    return (
                      <TableRow key={agent.agent_id}>
                        <TableCell>
                          <Text>{agent.agent_name}</Text>
                        </TableCell>
                        <TableCell>
                          <Tooltip title={agent.agent_id}>
                            <Button
                              size="xs"
                              variant="light"
                              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
                              onClick={() => setSelectedAgentId(agent.agent_id)}
                            >
                              {agent.agent_id.slice(0, 7)}...
                            </Button>
                          </Tooltip>
                        </TableCell>
                        <TableCell>
                          <Text>{formatNumberWithCommas(agent.spend, 4)}</Text>
                        </TableCell>
                        <TableCell>
                          <Badge size="xs" color="blue">
                            {agent.litellm_params?.model || "N/A"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Text>
                            {agent.created_at
                              ? new Date(agent.created_at).toLocaleDateString()
                              : "N/A"}
                          </Text>
                        </TableCell>
                        <TableCell>
                          {keyInfoMap[agent.agent_id]?.has_key ? (
                            <Badge color="green">Active</Badge>
                          ) : (
                            <Badge color="yellow">Needs Setup</Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          <DriftBadgeCell
                            drift={drift}
                            onClick={() => setSelectedAgentId(agent.agent_id)}
                          />
                        </TableCell>
                        {isAdmin && (
                          <TableCell>
                            <TableIconActionButton
                              variant="Delete"
                              onClick={() => handleDeleteClick(agent.agent_id, agent.agent_name)}
                            />
                          </TableCell>
                        )}
                      </TableRow>
                    );
                  })
                )}
              </TableBody>
            </Table>
          )}
        </Card>
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
