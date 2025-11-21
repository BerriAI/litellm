import { isAdminRole } from "@/utils/roles";
import { QuestionCircleOutlined } from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Button, Tab, TabGroup, TabList, TabPanel, TabPanels, Text, Title } from "@tremor/react";
import { Descriptions, Modal, Select, Tooltip, Typography } from "antd";
import React, { useEffect, useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";
import { deleteMCPServer, fetchMCPServers } from "../networking";
import { DataTable } from "../view_logs/table";
import CreateMCPServer from "./create_mcp_server";
import MCPConnect from "./mcp_connect";
import { mcpServerColumns } from "./mcp_server_columns";
import { MCPServerView } from "./mcp_server_view";
import { MCPServer, MCPServerProps, Team } from "./types";

const { Text: AntdText, Title: AntdTitle } = Typography;
const { Option } = Select;

const MCPServers: React.FC<MCPServerProps> = ({ accessToken, userRole, userID }) => {
  const {
    data: mcpServers,
    isLoading: isLoadingServers,
    refetch,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ["mcpServers"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchMCPServers(accessToken);
    },
    enabled: !!accessToken,
  }) as { data: MCPServer[]; isLoading: boolean; refetch: () => void; dataUpdatedAt: number };

  // Log allowed_tools from fetched servers
  React.useEffect(() => {
    if (mcpServers) {
      console.log("MCP Servers fetched:", mcpServers);
      mcpServers.forEach((server) => {
        console.log(`Server: ${server.server_name || server.server_id}`);
        console.log(`  allowed_tools:`, server.allowed_tools);
      });
    }
  }, [mcpServers]);

  // state
  const [serverIdToDelete, setServerToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedServerId, setSelectedServerId] = useState<string | null>(null);
  const [editServer, setEditServer] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<string>("all");
  const [selectedMcpAccessGroup, setSelectedMcpAccessGroup] = useState<string>("all");
  const [filteredServers, setFilteredServers] = useState<MCPServer[]>([]);
  const [isModalVisible, setModalVisible] = useState(false);
  const [isDeletingServer, setIsDeletingServer] = useState(false);
  const isInternalUser = userRole === "Internal User";

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

  // Get unique MCP access groups from all servers
  const uniqueMcpAccessGroups = React.useMemo(() => {
    if (!mcpServers) return [];
    return Array.from(
      new Set(
        mcpServers.flatMap((server) => server.mcp_access_groups).filter((group): group is string => group != null),
      ),
    );
  }, [mcpServers]);

  // Handle team filter change
  const handleTeamChange = (teamId: string) => {
    setSelectedTeam(teamId);
    filterServers(teamId, selectedMcpAccessGroup);
  };

  // Handle MCP access group filter change
  const handleMcpAccessGroupChange = (group: string) => {
    setSelectedMcpAccessGroup(group);
    filterServers(selectedTeam, group);
  };

  // Filtering logic for both team and access group
  const filterServers = (teamId: string, group: string) => {
    if (!mcpServers) return setFilteredServers([]);
    let filtered = mcpServers;
    if (teamId === "personal") {
      setFilteredServers([]);
      return;
    }
    if (teamId !== "all") {
      filtered = filtered.filter((server) => server.teams?.some((team) => team.team_id === teamId));
    }
    if (group !== "all") {
      filtered = filtered.filter((server) =>
        server.mcp_access_groups?.some((g: any) => (typeof g === "string" ? g === group : g && g.name === group)),
      );
    }
    setFilteredServers(filtered);
  };

  // Initial and effect-based filtering (trigger on query data updates)
  useEffect(() => {
    filterServers(selectedTeam, selectedMcpAccessGroup);
  }, [dataUpdatedAt]);

  const columns = React.useMemo(
    () =>
      mcpServerColumns(
        userRole ?? "",
        (serverId: string) => {
          setSelectedServerId(serverId);
          setEditServer(false);
        },
        (serverId: string) => {
          setSelectedServerId(serverId);
          setEditServer(true);
        },
        handleDelete,
      ),
    [userRole],
  );

  function handleDelete(server_id: string) {
    setServerToDelete(server_id);
    setIsDeleteModalOpen(true);
  }

  const confirmDelete = async () => {
    if (serverIdToDelete == null || accessToken == null) {
      return;
    }
    try {
      setIsDeletingServer(true);
      await deleteMCPServer(accessToken, serverIdToDelete);
      NotificationsManager.success("Deleted MCP Server successfully");
      refetch();
    } catch (error) {
      console.error("Error deleting the mcp server:", error);
    } finally {
      setIsDeletingServer(false);
      setIsDeleteModalOpen(false);
      setServerToDelete(null);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setServerToDelete(null);
  };

  // Find the server to delete from the servers list
  const serverToDelete = serverIdToDelete
    ? (mcpServers || []).find((server) => server.server_id === serverIdToDelete)
    : null;

  const handleCreateSuccess = (newMcpServer: MCPServer) => {
    setFilteredServers((prev) => [...prev, newMcpServer]);
    setModalVisible(false);
  };

  if (!accessToken || !userRole || !userID) {
    console.log("Missing required authentication parameters", { accessToken, userRole, userID });
    return <div className="p-6 text-center text-gray-500">Missing required authentication parameters.</div>;
  }

  const ServersTab = () =>
    selectedServerId ? (
      <MCPServerView
        mcpServer={
          filteredServers.find((server: MCPServer) => server.server_id === selectedServerId) || {
            server_id: "",
            server_name: "",
            alias: "",
            url: "",
            transport: "",
            auth_type: "",
            created_at: "",
            created_by: "",
            updated_at: "",
            updated_by: "",
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
        availableAccessGroups={uniqueMcpAccessGroups}
      />
    ) : (
      <div className="w-full h-full">
        <div className="w-full px-6">
          <div className="flex flex-col space-y-4">
            <div className="flex items-center justify-between bg-gray-50 rounded-lg p-4 border-2 border-gray-200">
              <div className="flex items-center gap-4">
                <Text className="text-lg font-semibold text-gray-900">Current Team:</Text>
                <Select value={selectedTeam} onChange={handleTeamChange} style={{ width: 300 }}>
                  <Option value="all">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <span className="font-medium">{isInternalUser ? "All Available Servers" : "All Servers"}</span>
                    </div>
                  </Option>
                  <Option value="personal">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="font-medium">Personal</span>
                    </div>
                  </Option>
                  {uniqueTeams.map((team) => (
                    <Option key={team.team_id} value={team.team_id}>
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="font-medium">{team.team_alias || team.team_id}</span>
                      </div>
                    </Option>
                  ))}
                </Select>
                <Text className="text-lg font-semibold text-gray-900 ml-6">
                  Access Group:
                  <Tooltip title="An MCP Access Group is a set of users or teams that have permission to access specific MCP servers. Use access groups to control and organize who can connect to which servers.">
                    <QuestionCircleOutlined style={{ marginLeft: 4, color: "#888" }} />
                  </Tooltip>
                </Text>
                <Select value={selectedMcpAccessGroup} onChange={handleMcpAccessGroupChange} style={{ width: 300 }}>
                  <Option value="all">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <span className="font-medium">All Access Groups</span>
                    </div>
                  </Option>
                  {uniqueMcpAccessGroups.map((group) => (
                    <Option key={group} value={group}>
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="font-medium">{group}</span>
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
    );

  return (
    <div className="w-full h-full p-6">
      <Modal
        open={isDeleteModalOpen}
        title="Delete MCP Server?"
        onOk={confirmDelete}
        okText={isDeletingServer ? "Deleting..." : "Delete"}
        onCancel={cancelDelete}
        cancelText="Cancel"
        cancelButtonProps={{ disabled: isDeletingServer }}
        okButtonProps={{ danger: true }}
        confirmLoading={isDeletingServer}
      >
        <div className="space-y-4">
          <AntdText>Are you sure you want to delete this MCP Server? This action cannot be undone.</AntdText>

          {serverToDelete && (
            <div className="mt-4 p-4 bg-red-50 rounded-lg border border-red-200">
              <AntdTitle level={5} className="mb-3 text-gray-900">
                Server Information
              </AntdTitle>
              <Descriptions column={1} size="small">
                {serverToDelete.server_name && (
                  <Descriptions.Item label={<span className="font-semibold text-gray-700">Server Name</span>}>
                    <AntdText className="text-sm">{serverToDelete.server_name}</AntdText>
                  </Descriptions.Item>
                )}
                {serverToDelete.alias && (
                  <Descriptions.Item label={<span className="font-semibold text-gray-700">Alias</span>}>
                    <AntdText className="text-sm">{serverToDelete.alias}</AntdText>
                  </Descriptions.Item>
                )}
                <Descriptions.Item label={<span className="font-semibold text-gray-700">Server ID</span>}>
                  <AntdText code className="text-sm">
                    {serverToDelete.server_id}
                  </AntdText>
                </Descriptions.Item>
                <Descriptions.Item label={<span className="font-semibold text-gray-700">URL</span>}>
                  <AntdText code className="text-sm">
                    {serverToDelete.url}
                  </AntdText>
                </Descriptions.Item>
              </Descriptions>
            </div>
          )}
        </div>
      </Modal>
      <CreateMCPServer
        userRole={userRole}
        accessToken={accessToken}
        onCreateSuccess={handleCreateSuccess}
        isModalVisible={isModalVisible}
        setModalVisible={setModalVisible}
        availableAccessGroups={uniqueMcpAccessGroups}
      />
      <Title>MCP Servers</Title>
      <Text className="text-tremor-content mt-2">Configure and manage your MCP servers</Text>
      {isAdminRole(userRole) && (
        <Button className="mt-4 mb-4" onClick={() => setModalVisible(true)}>
          + Add New MCP Server
        </Button>
      )}
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
