import { isAdminRole } from "@/utils/roles";
import { QuestionCircleOutlined } from "@ant-design/icons";
import { Button, Tab, TabGroup, TabList, TabPanel, TabPanels, Text, Title } from "@tremor/react";
import NewBadge from "../common_components/NewBadge";
import { Descriptions, Modal, Select, Tooltip, Typography } from "antd";
import React, { useEffect, useState, useMemo, useCallback } from "react";
import { useMCPServers } from "../../app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPServerHealth } from "../../app/(dashboard)/hooks/mcpServers/useMCPServerHealth";
import NotificationsManager from "../molecules/notifications_manager";
import { deleteMCPServer } from "../networking";
import { MCPSubmissionsTab } from "./MCPSubmissionsTab";
import { DataTable } from "../view_logs/table";
import CreateMCPServer from "./create_mcp_server";
import MCPConnect from "./mcp_connect";
import { mcpServerColumns } from "./mcp_server_columns";
import { MCPServerView } from "./mcp_server_view";
import { DiscoverableMCPServer, MCPServer, MCPServerProps, Team } from "./types";
import MCPSemanticFilterSettings from "../Settings/AdminSettings/MCPSemanticFilterSettings/MCPSemanticFilterSettings";
import MCPNetworkSettings from "./MCPNetworkSettings";
import MCPDiscovery from "./mcp_discovery";
import { ByokCredentialModal } from "./ByokCredentialModal";

const { Text: AntdText, Title: AntdTitle } = Typography;
const EDIT_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-edit-state";

const { Option } = Select;

const MCPServers: React.FC<MCPServerProps> = ({ accessToken, userRole, userID }) => {
  const { data: mcpServers, isLoading: isLoadingServers, refetch } = useMCPServers();

  // Fetch health status for all servers
  const { data: healthStatuses, isLoading: isLoadingHealth, recheckServerHealth, recheckingServerIds } = useMCPServerHealth();

  // Merge health status data into servers
  const serversWithHealth = useMemo(() => {
    if (!mcpServers) return [];
    if (!healthStatuses) return mcpServers;

    const healthMap = new Map(healthStatuses.map((h) => [h.server_id, h.status]));

    return mcpServers.map((server) => {
      const healthStatus = healthMap.get(server.server_id);
      return {
        ...server,
        status: healthStatus
          ? (healthStatus as "healthy" | "unhealthy" | "unknown")
          : server.status,
      };
    });
  }, [mcpServers, healthStatuses]);

  // state
  const [serverIdToDelete, setServerToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedServerId, setSelectedServerId] = useState<string | null>(null);
  const [editServer, setEditServer] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<string>("all");
  const [selectedMcpAccessGroup, setSelectedMcpAccessGroup] = useState<string>("all");
  const [filteredServers, setFilteredServers] = useState<MCPServer[]>([]);
  const [isModalVisible, setModalVisible] = useState(false);
  const [isDiscoveryVisible, setDiscoveryVisible] = useState(false);
  const [prefillData, setPrefillData] = useState<DiscoverableMCPServer | null>(null);
  const [isDeletingServer, setIsDeletingServer] = useState(false);
  const [byokModalServer, setByokModalServer] = useState<MCPServer | null>(null);
  const isInternalUser = userRole === "Internal User";

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const stored = window.sessionStorage.getItem(EDIT_OAUTH_UI_STATE_KEY);
      if (!stored) {
        return;
      }
      const parsed = JSON.parse(stored);
      if (parsed?.serverId) {
        setSelectedServerId(parsed.serverId);
        setEditServer(true);
      }
    } catch (err) {
      console.error("Failed to restore MCP edit view state", err);
    }
  }, []);

  // Get unique teams from all servers
  const uniqueTeams = React.useMemo(() => {
    if (!serversWithHealth) return [];
    const teamsSet = new Set<string>();
    const uniqueTeamsArray: Team[] = [];
    serversWithHealth.forEach((server: MCPServer) => {
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
  }, [serversWithHealth]);

  // Get unique MCP access groups from all servers
  const uniqueMcpAccessGroups = React.useMemo(() => {
    if (!serversWithHealth) return [];
    return Array.from(
      new Set(
        serversWithHealth.flatMap((server) => server.mcp_access_groups).filter((group): group is string => group != null),
      ),
    );
  }, [serversWithHealth]);

  // Filtering logic for both team and access group
  const filterServers = useCallback((teamId: string, group: string) => {
    if (!serversWithHealth) return setFilteredServers([]);
    let filtered = serversWithHealth;
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
    const sorted = [...filtered].sort((a, b) => {
      if (!a.created_at && !b.created_at) return 0;
      if (!a.created_at) return 1;
      if (!b.created_at) return -1;
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });
    setFilteredServers(sorted);
  }, [serversWithHealth]);

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

  // Initial and effect-based filtering (trigger on query data updates and health data updates)
  useEffect(() => {
    filterServers(selectedTeam, selectedMcpAccessGroup);
  }, [serversWithHealth, selectedTeam, selectedMcpAccessGroup, filterServers]);

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
        isLoadingHealth,
        (server: MCPServer) => setByokModalServer(server),
        recheckServerHealth,
        recheckingServerIds,
      ),
    [userRole, isLoadingHealth, recheckServerHealth, recheckingServerIds],
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

  // Memoize the selected server to prevent unnecessary re-renders
  const selectedServer = React.useMemo(() => {
    return filteredServers.find((server: MCPServer) => server.server_id === selectedServerId) || {
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
    };
  }, [filteredServers, selectedServerId]);

  // Memoize the onBack callback to prevent unnecessary re-renders
  const handleBack = React.useCallback(() => {
    setEditServer(false);
    setSelectedServerId(null);
    refetch();
  }, [refetch]);

  if (!accessToken || !userRole || !userID) {
    console.log("Missing required authentication parameters", { accessToken, userRole, userID });
    return <div className="p-6 text-center text-gray-500">Missing required authentication parameters.</div>;
  }

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
          <AntdText className="text-gray-600">This action is permanent and cannot be undone. All associated configurations will be removed.</AntdText>

          {serverToDelete && (
            <div className="mt-3 p-4 bg-gray-50 rounded-lg border border-gray-200">
              <Descriptions column={1} size="small" colon={false}>
                {serverToDelete.server_name && (
                  <Descriptions.Item label={<span className="text-gray-500 text-sm">Name</span>}>
                    <AntdText strong className="text-sm">{serverToDelete.server_name}</AntdText>
                  </Descriptions.Item>
                )}
                <Descriptions.Item label={<span className="text-gray-500 text-sm">ID</span>}>
                  <AntdText code className="text-xs">
                    {serverToDelete.server_id}
                  </AntdText>
                </Descriptions.Item>
                {serverToDelete.url && (
                  <Descriptions.Item label={<span className="text-gray-500 text-sm">URL</span>}>
                    <AntdText code className="text-xs break-all">
                      {serverToDelete.url}
                    </AntdText>
                  </Descriptions.Item>
                )}
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
        prefillData={prefillData}
        onBackToDiscovery={() => {
          setModalVisible(false);
          setPrefillData(null);
          setDiscoveryVisible(true);
        }}
      />
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <Title>MCP Servers</Title>
            {filteredServers.length > 0 && (
              <span className="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 border border-gray-200">
                {filteredServers.length}
              </span>
            )}
          </div>
          <Text className="text-tremor-content mt-1">Configure and manage your MCP servers</Text>
        </div>
        <div className="flex items-center gap-2">
          {isAdminRole(userRole) && (
            <Button className="flex-shrink-0" onClick={() => setDiscoveryVisible(true)}>
              + Add New MCP Server
            </Button>
          )}
          {!isAdminRole(userRole) && (
            <Button
              className="flex-shrink-0"
              onClick={() => {
                setPrefillData(null);
                setModalVisible(true);
              }}
              variant="secondary"
            >
              + Submit MCP Server
            </Button>
          )}
        </div>
      </div>
      <MCPDiscovery
        isVisible={isDiscoveryVisible}
        onClose={() => setDiscoveryVisible(false)}
        onSelectServer={(server: DiscoverableMCPServer) => {
          setPrefillData(server);
          setDiscoveryVisible(false);
          setModalVisible(true);
        }}
        onCustomServer={() => {
          setPrefillData(null);
          setDiscoveryVisible(false);
          setModalVisible(true);
        }}
        accessToken={accessToken}
      />
      <TabGroup className="w-full h-full">
        <TabList className="flex justify-between mt-2 w-full items-center">
          <div className="flex">
            <Tab>All Servers</Tab>
            <Tab>Connect</Tab>
            <Tab>Semantic Filter</Tab>
            <Tab>Network Settings</Tab>
            {isAdminRole(userRole) && <Tab><span className="flex items-center gap-2">Submitted MCPs <NewBadge /></span></Tab>}
          </div>
        </TabList>
        <TabPanels>
          <TabPanel>
            {selectedServerId ? (
              <MCPServerView
                key={selectedServerId}
                mcpServer={selectedServer}
                onBack={handleBack}
                isProxyAdmin={isAdminRole(userRole)}
                isEditing={editServer}
                accessToken={accessToken}
                userID={userID}
                userRole={userRole}
                availableAccessGroups={uniqueMcpAccessGroups}
              />
            ) : (
              <div className="w-full h-full">
                <div className="w-full">
                  <div className="flex flex-col space-y-4">
                    <div className="flex items-center gap-6 bg-white rounded-lg px-4 py-3 border border-gray-200">
                      <div className="flex items-center gap-2">
                        <Text className="text-sm font-medium text-gray-600 whitespace-nowrap">Team</Text>
                        <Select value={selectedTeam} onChange={handleTeamChange} style={{ width: 220 }} size="middle">
                          <Option value="all">
                            <span className="font-medium">{isInternalUser ? "All Available Servers" : "All Servers"}</span>
                          </Option>
                          <Option value="personal">
                            <span className="font-medium">Personal</span>
                          </Option>
                          {uniqueTeams.map((team) => (
                            <Option key={team.team_id} value={team.team_id}>
                              <span className="font-medium">{team.team_alias || team.team_id}</span>
                            </Option>
                          ))}
                        </Select>
                      </div>
                      <div className="h-6 w-px bg-gray-200"></div>
                      <div className="flex items-center gap-2">
                        <Text className="text-sm font-medium text-gray-600 whitespace-nowrap">
                          Access Group
                          <Tooltip title="An MCP Access Group is a set of users or teams that have permission to access specific MCP servers. Use access groups to control and organize who can connect to which servers.">
                            <QuestionCircleOutlined style={{ marginLeft: 4, color: "#9ca3af" }} />
                          </Tooltip>
                        </Text>
                        <Select value={selectedMcpAccessGroup} onChange={handleMcpAccessGroupChange} style={{ width: 220 }} size="middle">
                          <Option value="all">
                            <span className="font-medium">All Access Groups</span>
                          </Option>
                          {uniqueMcpAccessGroups.map((group) => (
                            <Option key={group} value={group}>
                              <span className="font-medium">{group}</span>
                            </Option>
                          ))}
                        </Select>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="w-full mt-6">
                  <DataTable
                    data={filteredServers}
                    columns={columns}
                    renderSubComponent={() => <div></div>}
                    getRowCanExpand={() => false}
                    isLoading={isLoadingServers}
                    noDataMessage="No MCP servers configured. Click '+ Add New MCP Server' to get started."
                    loadingMessage="Loading MCP servers..."
                    enableSorting={true}
                  />
                </div>
              </div>
            )}
          </TabPanel>
          <TabPanel>
            <MCPConnect />
          </TabPanel>
          <TabPanel>
            <MCPSemanticFilterSettings accessToken={accessToken} />
          </TabPanel>
          <TabPanel>
            <MCPNetworkSettings accessToken={accessToken} />
          </TabPanel>
          {isAdminRole(userRole) && (
            <TabPanel>
              <MCPSubmissionsTab accessToken={accessToken} />
            </TabPanel>
          )}
        </TabPanels>
      </TabGroup>

      {byokModalServer && (
        <ByokCredentialModal
          server={byokModalServer}
          open={!!byokModalServer}
          onClose={() => setByokModalServer(null)}
          onSuccess={(_serverId) => {
            refetch();
            setByokModalServer(null);
          }}
          accessToken={accessToken || ""}
        />
      )}
    </div>
  );
};

export default MCPServers;
