import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  Modal,
  Tooltip,
  message,
  Select,
  Space,
  Typography
} from "antd";
import {
  TabPanel,
  TabPanels,
  TabGroup,
  TabList,
  Tab,
} from "@tremor/react";
import { LinkIcon } from "lucide-react";

import {
  Grid,
  Col,
  Title,
  Text,
} from "@tremor/react";
import { DataTable } from "../view_logs/table";
import { mcpServerColumns } from "./mcp_server_columns";

import {
  deleteMCPServer,
  fetchMCPServers,
} from "../networking";
import {
  MCPServer,
  MCPServerProps,
  Team,
  handleAuth,
  handleTransport,
} from "./types";
import { isAdminRole } from "@/utils/roles";
import { MCPServerView } from "./mcp_server_view";
import CreateMCPServer from "./create_mcp_server";
import MCPConnect from "./mcp_connect";

const { Option } = Select;

interface DeleteModalProps {
  isModalOpen: boolean;
  title: string;
  confirmDelete: () => void;
  cancelDelete: () => void;
}

const DeleteModal: React.FC<DeleteModalProps> = ({
  isModalOpen,
  title,
  confirmDelete,
  cancelDelete,
}) => {
  if (!isModalOpen) return null;

  return (
    <Modal
      open={isModalOpen}
      onOk={confirmDelete}
      okType="danger"
      onCancel={cancelDelete}
    >
      <Grid numItems={1} className="gap-2 w-full">
        <Title>{title}</Title>
        <Col numColSpan={1}>
          <p>Are you sure you want to delete this MCP Server?</p>
        </Col>
      </Grid>
    </Modal>
  );
};

const MCPServers: React.FC<MCPServerProps> = ({
  accessToken,
  userRole,
  userID,
}) => {
  // Query to fetch MCP tools
  const {
    data: mcpServers,
    isLoading: isLoadingServers,
    refetch,
  } = useQuery({
    queryKey: ["mcpServers"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchMCPServers(accessToken);
    },
    enabled: !!accessToken,
  }) as { data: MCPServer[]; isLoading: boolean; refetch: () => void };

  const createMCPServer = (newMcpServer: any) => {
    refetch();
  };

  // state
  const [serverIdToDelete, setServerToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedServerId, setSelectedServerId] = useState<string | null>(null);
  const [editServer, setEditServer] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<string>("all");
  const [filteredServers, setFilteredServers] = useState<MCPServer[]>([]);
  const [currentTeam, setCurrentTeam] = useState<string>("personal");
  const [modelViewMode, setModelViewMode] = useState<string>("all");

  // Get unique teams from all servers
  const uniqueTeams = React.useMemo(() => {
    if (!mcpServers) return [];
    const teamsSet = new Set<string>();
    const uniqueTeamsArray: Team[] = [];
    
    mcpServers.forEach((server: MCPServer) => {
      if (server.teams) {
        server.teams.forEach((team: Team) => {
          const teamKey = team.team_id;
          if (!teamsSet.has(teamKey)) {
            teamsSet.add(teamKey);
            uniqueTeamsArray.push(team);
          }
        });
      }
    });
    return uniqueTeamsArray;
  }, [mcpServers]);

  // Handle team filter change
  const handleTeamChange = (teamId: string) => {
    setSelectedTeam(teamId);
    setCurrentTeam(teamId);
    
    if (!mcpServers) return;
    
    if (teamId === "all") {
      setFilteredServers(mcpServers);
    } else {
      const filtered = mcpServers.filter(server => 
        server.teams?.some(team => team.team_id === teamId)
      );
      setFilteredServers(filtered);
    }
  };

  React.useEffect(() => {
    if (mcpServers) {
      setFilteredServers(mcpServers);
    }
  }, [mcpServers]);

  const columns = React.useMemo(
    () =>
      mcpServerColumns(
        userRole,
        (serverId: string) => setSelectedServerId(serverId),
        (serverId: string) => {
          setSelectedServerId(serverId);
          setEditServer(true);
        },
        handleDelete
      ),
    [userRole]
  );

  function handleDelete(server_id: string) {
    // Set the team to delete and open the confirmation modal
    setServerToDelete(server_id);
    setIsDeleteModalOpen(true);
  }

  const confirmDelete = async () => {
    if (serverIdToDelete == null || accessToken == null) {
      return;
    }

    try {
      await deleteMCPServer(accessToken, serverIdToDelete);
      // Successfully completed the deletion. Update the state to trigger a rerender.
      message.success("Deleted MCP Server successfully");
      refetch();
    } catch (error) {
      console.error("Error deleting the mcp server:", error);
      // Handle any error situations, such as displaying an error message to the user.
    }

    // Close the confirmation modal and reset the serverToDelete
    setIsDeleteModalOpen(false);
    setServerToDelete(null);
  };

  const cancelDelete = () => {
    // Close the confirmation modal and reset the serverToDelete
    setIsDeleteModalOpen(false);
    setServerToDelete(null);
  };

  // Check if user is internal or admin
  const canAccessMCPServers = isAdminRole(userRole!) || (userRole && userRole.includes('internal'));

  if (!accessToken || !userRole || !userID) {
    return (
      <div className="p-6 text-center text-gray-500">
        Missing required authentication parameters.
      </div>
    );
  }

  const ServersTab = () => (
    selectedServerId ? (
      <MCPServerView
        mcpServer={
          mcpServers.find(
            (server: MCPServer) => server.server_id === selectedServerId
          ) || {
            server_id: '',
            alias: '',
            url: '',
            transport: '',
            spec_version: '',
            auth_type: '',
            created_at: '',
            created_by: '',
            updated_at: '',
            updated_by: '',
          }
        }
        onBack={() => {
          setEditServer(false);
          setSelectedServerId(null);
          refetch();
        }}
        isProxyAdmin={isAdminRole(userRole)}
        isEditing={editServer}
        accessToken={accessToken}
        userID={userID}
        userRole={userRole}
      />
    ) : (
      <div className="w-full h-full">
        <div className="flex justify-between items-center mb-6 px-6">
          <div>
            <Title>MCP Servers</Title>
            <Text className="text-tremor-content">
              Configure and manage your MCP servers
            </Text>
          </div>
          {isAdminRole(userRole) && (
            <CreateMCPServer
              userRole={userRole}
              accessToken={accessToken}
              onCreateSuccess={createMCPServer}
            />
          )}
        </div>
        <div className="w-full px-6">
          <div className="flex flex-col space-y-4">
            <div className="flex items-center justify-between bg-gray-50 rounded-lg p-4 border-2 border-gray-200">
              <div className="flex items-center gap-4">
                <Text className="text-lg font-semibold text-gray-900">Current Team:</Text>
                <Select 
                  value={currentTeam}
                  onChange={handleTeamChange}
                  style={{ width: 300 }}
                >
                  <Option value="personal">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <span className="font-medium">All Servers</span>
                    </div>
                  </Option>
                  {uniqueTeams.map((team) => (
                    <Option key={team.team_id} value={team.team_id}>
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="font-medium">
                          {team.team_alias || team.team_id}
                        </span>
                      </div>
                    </Option>
                  ))}
                </Select>
              </div>
            </div>
          </div>
        </div>
        <div className="w-full px-6 mt-6">
          <DataTable
            data={filteredServers}
            columns={columns}
            renderSubComponent={() => <div></div>}
            getRowCanExpand={() => false}
            isLoading={isLoadingServers}
            noDataMessage="No MCP servers configured"
          />
        </div>
      </div>
    )
  );

  return (
    <div className="w-full h-full">
      <DeleteModal
        isModalOpen={isDeleteModalOpen}
        title="Delete MCP Server"
        confirmDelete={confirmDelete}
        cancelDelete={cancelDelete}
      />
      <TabGroup className="w-full h-full">
        <TabList className="flex justify-between mt-2 w-full items-center">
          <div className="flex">
            <Tab>All Servers</Tab>
            <Tab>Connect</Tab>
          </div>
        </TabList>
        <TabPanels>
          <TabPanel>
            <ServersTab />
          </TabPanel>
          <TabPanel>
            <MCPConnect />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default MCPServers;
