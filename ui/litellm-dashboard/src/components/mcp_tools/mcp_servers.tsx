import { isAdminRole } from "@/utils/roles";
import { QuestionCircleOutlined, SearchOutlined } from "@ant-design/icons";
import { Button, Tab, TabGroup, TabList, TabPanel, TabPanels, Text, Title } from "@tremor/react";
import NewBadge from "../common_components/NewBadge";
import { Descriptions, Empty, Input, Modal, Select, Spin, Tooltip, Typography } from "antd";
import React, { useEffect, useState, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { useMCPServers } from "../../app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPServerHealth } from "../../app/(dashboard)/hooks/mcpServers/useMCPServerHealth";
import NotificationsManager from "../molecules/notifications_manager";
import { deleteMCPServer } from "../networking";
import { MCPSubmissionsTab } from "./MCPSubmissionsTab";
import { MCPToolsetsTab } from "./MCPToolsetsTab";
import CreateMCPServer from "./create_mcp_server";
import MCPConnect from "./mcp_connect";
import MCPServerCard from "./MCPServerCard";
import { MCPServerView } from "./mcp_server_view";
import type { DiscoverableMCPServer, MCPServer, MCPServerProps, MCPUserEnvVarsStatus, Team } from "./types";
import MCPSemanticFilterSettings from "../Settings/AdminSettings/MCPSemanticFilterSettings/MCPSemanticFilterSettings";
import MCPNetworkSettings from "./MCPNetworkSettings";
import MCPDiscovery from "./mcp_discovery";
import { ByokCredentialModal } from "./ByokCredentialModal";
import { getSecureItem } from "@/utils/secureStorage";
import { TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import UserEnvVarsModal from "./UserEnvVarsModal";
import { listMCPUserEnvVarStatus } from "../networking";

type SortKey = "created_desc" | "updated_desc" | "name_asc" | "health";

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "created_desc", label: "Recently created" },
  { value: "updated_desc", label: "Recently updated" },
  { value: "name_asc", label: "Name (A→Z)" },
  { value: "health", label: "Health (unhealthy first)" },
];

const HEALTH_RANK: Record<string, number> = {
  unhealthy: 0,
  unknown: 1,
  healthy: 2,
};

const compareServers = (a: MCPServer, b: MCPServer, sort: SortKey): number => {
  switch (sort) {
    case "name_asc": {
      const nameA = (a.server_name || a.alias || a.server_id).toLowerCase();
      const nameB = (b.server_name || b.alias || b.server_id).toLowerCase();
      return nameA.localeCompare(nameB);
    }
    case "updated_desc": {
      const ta = a.updated_at ? new Date(a.updated_at).getTime() : 0;
      const tb = b.updated_at ? new Date(b.updated_at).getTime() : 0;
      return tb - ta;
    }
    case "health": {
      const ra = HEALTH_RANK[a.status ?? "unknown"] ?? 1;
      const rb = HEALTH_RANK[b.status ?? "unknown"] ?? 1;
      if (ra !== rb) return ra - rb;
      const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
      const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
      return tb - ta;
    }
    case "created_desc":
    default: {
      const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
      const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
      return tb - ta;
    }
  }
};

const { Text: AntdText, Title: AntdTitle } = Typography;
const EDIT_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-edit-state";

// Server id stashed by the Tools tab before an OBO OAuth redirect, read once at
// mount so the redirect returns straight to that server's Tools tab.
const readToolsOAuthServerId = (): string | null => {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const stored = getSecureItem(TOOLS_OAUTH_UI_STATE_KEY);
    if (!stored) {
      return null;
    }
    return JSON.parse(stored)?.serverId ?? null;
  } catch {
    return null;
  }
};

const { Option } = Select;

const MCPServers: React.FC<MCPServerProps> = ({ accessToken, userRole, userID }) => {
  const { data: mcpServers, isLoading: isLoadingServers, refetch } = useMCPServers();

  // Fetch health status for all servers
  const {
    data: healthStatuses,
    isLoading: isLoadingHealth,
    recheckServerHealth,
    recheckingServerIds,
  } = useMCPServerHealth();

  // Merge health status data into servers
  const serversWithHealth = useMemo(() => {
    if (!mcpServers) return [];
    if (!healthStatuses) return mcpServers;

    const healthMap = new Map(healthStatuses.map((h) => [h.server_id, h.status]));

    return mcpServers.map((server) => {
      const healthStatus = healthMap.get(server.server_id);
      return {
        ...server,
        status: healthStatus ? (healthStatus as "healthy" | "unhealthy" | "unknown") : server.status,
      };
    });
  }, [mcpServers, healthStatuses]);

  // state
  const [serverIdToDelete, setServerToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  // Server whose Tools tab should be reopened after an OBO OAuth redirect; read
  // once from sessionStorage so the restored server selection is correct on the
  // first render. Cleared when the user navigates back to the list (handleBack)
  // so a later visit to the same server defaults to Overview, not the Tools tab.
  const [toolsTabServerId, setToolsTabServerId] = useState<string | null>(readToolsOAuthServerId);
  const [selectedServerId, setSelectedServerId] = useState<string | null>(toolsTabServerId);
  const [editServer, setEditServer] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<string>("all");
  const [selectedMcpAccessGroup, setSelectedMcpAccessGroup] = useState<string>("all");
  const [filteredServers, setFilteredServers] = useState<MCPServer[]>([]);
  const [isModalVisible, setModalVisible] = useState(false);
  const [isDiscoveryVisible, setDiscoveryVisible] = useState(false);
  const [prefillData, setPrefillData] = useState<DiscoverableMCPServer | null>(null);
  const [isDeletingServer, setIsDeletingServer] = useState(false);
  const [byokModalServer, setByokModalServer] = useState<MCPServer | null>(null);
  // Per-user env-var fill modal target + deep-link source captured once from the URL.
  const [envVarsModalServer, setEnvVarsModalServer] = useState<MCPServer | null>(null);
  const [deepLinkServerId, setDeepLinkServerId] = useState<string | null>(() =>
    typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("fill_env_vars"),
  );
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [sortKey, setSortKey] = useState<SortKey>("created_desc");
  const isInternalUser = userRole === "Internal User";

  // Single bulk fetch of this user's per-server env-var status. Drives the
  // red "N user fields missing" footer on each card with no per-row request.
  const { data: envVarStatuses, refetch: refetchEnvVarStatus } = useQuery<MCPUserEnvVarsStatus[]>({
    queryKey: ["mcpUserEnvVarStatus"],
    queryFn: () => listMCPUserEnvVarStatus(accessToken!),
    enabled: !!accessToken,
  });

  // Per-server list of per-user fields this user still needs to fill in.
  const missingFieldsByServer = useMemo(() => {
    const map: Record<string, string[]> = {};
    for (const status of envVarStatuses ?? []) {
      map[status.server_id] = (status.required ?? []).filter((spec) => !spec.is_set).map((spec) => spec.name);
    }
    return map;
  }, [envVarStatuses]);

  // Deep-link via ?fill_env_vars=<server_id> — the link users follow from the
  // friendly error the proxy returns when a per-user var is missing. The id is
  // captured into state above and resolved to a server below; here we only strip
  // the param so a refresh doesn't reopen the modal.
  useEffect(() => {
    if (!deepLinkServerId || typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    if (!params.has("fill_env_vars")) return;
    params.delete("fill_env_vars");
    const newSearch = params.toString();
    const newUrl = window.location.pathname + (newSearch ? `?${newSearch}` : "") + window.location.hash;
    window.history.replaceState({}, "", newUrl);
  }, [deepLinkServerId]);

  const deepLinkServer = useMemo(
    () => (deepLinkServerId ? serversWithHealth.find((s) => s.server_id === deepLinkServerId) ?? null : null),
    [deepLinkServerId, serversWithHealth],
  );
  const activeEnvVarsServer = envVarsModalServer ?? deepLinkServer;

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const stored = getSecureItem(EDIT_OAUTH_UI_STATE_KEY);
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

  // The restored server id was consumed by the initializer above; remove the
  // one-shot sessionStorage key so a full page reload doesn't reopen the Tools
  // tab (removeItem only, no setState).
  useEffect(() => {
    if (typeof window !== "undefined") {
      try {
        window.sessionStorage.removeItem(TOOLS_OAUTH_UI_STATE_KEY);
      } catch {
        // ignore storage errors
      }
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
        serversWithHealth
          .flatMap((server) => server.mcp_access_groups)
          .filter((group): group is string => group != null),
      ),
    );
  }, [serversWithHealth]);

  // Filtering logic for both team and access group
  const filterServers = useCallback(
    (teamId: string, group: string) => {
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
    },
    [serversWithHealth],
  );

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

  // Search + sort layer applied on top of the team/access-group filters.
  const displayedServers = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    const matches = q
      ? filteredServers.filter((s) => {
          const name = (s.server_name || "").toLowerCase();
          const alias = (s.alias || "").toLowerCase();
          const url = (s.url || "").toLowerCase();
          const id = s.server_id.toLowerCase();
          return name.includes(q) || alias.includes(q) || url.includes(q) || id.includes(q);
        })
      : filteredServers;
    return [...matches].sort((a, b) => compareServers(a, b, sortKey));
  }, [filteredServers, searchQuery, sortKey]);

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
      // If the user is currently viewing the detail page of the server they
      // just deleted, return them to the All Servers list. Otherwise the
      // detail view would stay mounted, fall back to an empty stub server,
      // and show a phantom "Unnamed Server" page.
      if (selectedServerId === serverIdToDelete) {
        setEditServer(false);
        setSelectedServerId(null);
      }
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
    refetch();
  };

  // Memoize the selected server to prevent unnecessary re-renders
  const selectedServer = React.useMemo(() => {
    return (
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
    );
  }, [filteredServers, selectedServerId]);

  // Memoize the onBack callback to prevent unnecessary re-renders
  const handleBack = React.useCallback(() => {
    setEditServer(false);
    setSelectedServerId(null);
    // Drop the post-redirect one-shot so re-selecting that server opens Overview.
    setToolsTabServerId(null);
    refetch();
  }, [refetch]);

  if (!accessToken || !userRole || !userID) {
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
          <AntdText className="text-gray-600">
            This action is permanent and cannot be undone. All associated configurations will be removed.
          </AntdText>

          {serverToDelete && (
            <div className="mt-3 p-4 bg-gray-50 rounded-lg border border-gray-200">
              <Descriptions column={1} size="small" colon={false}>
                {serverToDelete.server_name && (
                  <Descriptions.Item label={<span className="text-gray-500 text-sm">Name</span>}>
                    <AntdText strong className="text-sm">
                      {serverToDelete.server_name}
                    </AntdText>
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
        userID={userID}
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
            <Button className="shrink-0" onClick={() => setDiscoveryVisible(true)}>
              + Add New MCP Server
            </Button>
          )}
          {!isAdminRole(userRole) && (
            <Button
              className="shrink-0"
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
            <Tab>Toolsets</Tab>
            <Tab>Connect</Tab>
            {isAdminRole(userRole) && <Tab>Semantic Filter</Tab>}
            {isAdminRole(userRole) && <Tab>Network Settings</Tab>}
            {isAdminRole(userRole) && (
              <Tab>
                <span className="flex items-center gap-2">
                  Submitted MCPs <NewBadge />
                </span>
              </Tab>
            )}
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
                initialTabIndex={selectedServerId === toolsTabServerId ? 1 : 0}
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
                            <span className="font-medium">
                              {isInternalUser ? "All Available Servers" : "All Servers"}
                            </span>
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
                        <Select
                          value={selectedMcpAccessGroup}
                          onChange={handleMcpAccessGroupChange}
                          style={{ width: 220 }}
                          size="middle"
                        >
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
                <div className="mt-4 flex flex-wrap items-center gap-3">
                  <Input
                    allowClear
                    prefix={<SearchOutlined className="text-gray-400" />}
                    placeholder="Search by name, alias, URL, or ID"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{ maxWidth: 320 }}
                  />
                  <div className="flex items-center gap-2">
                    <Text className="whitespace-nowrap text-sm font-medium text-gray-600">Sort</Text>
                    <Select
                      value={sortKey}
                      onChange={(v: SortKey) => setSortKey(v)}
                      style={{ width: 220 }}
                      size="middle"
                    >
                      {SORT_OPTIONS.map((opt) => (
                        <Option key={opt.value} value={opt.value}>
                          {opt.label}
                        </Option>
                      ))}
                    </Select>
                  </div>
                  <div className="ml-auto text-xs text-gray-500">
                    {displayedServers.length} of {filteredServers.length} servers
                  </div>
                </div>
                <div className="mt-4 w-full">
                  {isLoadingServers ? (
                    <div className="flex items-center justify-center rounded-lg border border-dashed border-gray-200 bg-white p-12">
                      <Spin tip="Loading MCP servers..." />
                    </div>
                  ) : displayedServers.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-gray-200 bg-white p-12">
                      <Empty
                        description={
                          filteredServers.length === 0
                            ? "No MCP servers configured. Click '+ Add New MCP Server' to get started."
                            : "No servers match the current filters or search."
                        }
                      />
                    </div>
                  ) : (
                    <div
                      data-testid="mcp-servers-grid"
                      className="grid auto-rows-fr grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3"
                    >
                      {displayedServers.map((server) => (
                        <MCPServerCard
                          key={server.server_id}
                          server={server}
                          missingUserFields={missingFieldsByServer[server.server_id]}
                          isLoadingHealth={isLoadingHealth}
                          isRechecking={recheckingServerIds?.has(server.server_id)}
                          onClick={() => {
                            setSelectedServerId(server.server_id);
                            setEditServer(true);
                          }}
                          onRecheckHealth={
                            recheckServerHealth ? () => recheckServerHealth(server.server_id) : undefined
                          }
                          onByokConnect={server.is_byok ? () => setByokModalServer(server) : undefined}
                          onOpenFillFields={() => setEnvVarsModalServer(server)}
                          onDelete={isAdminRole(userRole) ? () => handleDelete(server.server_id) : undefined}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </TabPanel>
          <TabPanel>
            <MCPToolsetsTab accessToken={accessToken} userRole={userRole} />
          </TabPanel>
          <TabPanel>
            <MCPConnect />
          </TabPanel>
          {isAdminRole(userRole) && (
            <TabPanel>
              <MCPSemanticFilterSettings accessToken={accessToken} />
            </TabPanel>
          )}
          {isAdminRole(userRole) && (
            <TabPanel>
              <MCPNetworkSettings accessToken={accessToken} />
            </TabPanel>
          )}
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

      {/* Per-user env-var fill modal — backed by /v1/mcp/server/{id}/user-env-vars */}
      <UserEnvVarsModal
        server={activeEnvVarsServer}
        open={!!activeEnvVarsServer}
        accessToken={accessToken}
        onClose={() => {
          setEnvVarsModalServer(null);
          setDeepLinkServerId(null);
        }}
        onSaved={() => {
          // Refresh the bulk status so the red "N user fields missing" footer
          // on each card clears once the user has filled in their values.
          refetchEnvVarStatus();
        }}
      />
    </div>
  );
};

export default MCPServers;
