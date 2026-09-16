import { isAdminRole } from "@/utils/roles";
import { CircleHelp, Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import React, { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPServerHealth } from "@/app/(dashboard)/hooks/mcpServers/useMCPServerHealth";
import { toast } from "@/lib/toast";
import { deleteMCPServer } from "@/components/networking";
import { MCPSubmissionsTab } from "./MCPSubmissionsTab";
import { MCPToolsetsTab } from "./MCPToolsetsTab";
import CreateMCPServer from "./CreateMCPServer";
import ImportMCPServers from "./ImportMCPServers";
import MCPConnect from "./mcp_connect";
import MCPServerCard from "./MCPServerCard";
import { MCPServerView } from "./mcp_server_view";
import type {
  DiscoverableMCPServer,
  MCPServer,
  MCPServerProps,
  MCPUserEnvVarsStatus,
  Team,
} from "@/components/mcp_tools/types";
import MCPSemanticFilterSettings from "@/components/Settings/AdminSettings/MCPSemanticFilterSettings/MCPSemanticFilterSettings";
import MCPToolSearchSettings from "@/components/Settings/AdminSettings/MCPToolSearchSettings/MCPToolSearchSettings";
import MCPNetworkSettings from "./MCPNetworkSettings";
import MCPDiscovery from "./mcp_discovery";
import { ByokCredentialModal } from "@/components/mcp_tools/ByokCredentialModal";
import UserEnvVarsModal from "./UserEnvVarsModal";
import { listMCPUserEnvVarStatus } from "@/components/networking";
import { type McpServerSortKey, useMcpServersUrlState } from "./useMcpServersUrlState";

const SORT_OPTIONS: { value: McpServerSortKey; label: string }[] = [
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

const compareServers = (a: MCPServer, b: MCPServer, sort: McpServerSortKey): number => {
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

const createdAtDesc = (a: MCPServer, b: MCPServer): number => {
  if (!a.created_at && !b.created_at) return 0;
  if (!a.created_at) return 1;
  if (!b.created_at) return -1;
  return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
};

type AccessGroupEntry = string | { name?: string } | null;

const accessGroupName = (group: AccessGroupEntry) => (typeof group === "string" ? group : group?.name);

const inAccessGroup = (server: MCPServer, group: string): boolean =>
  (server.mcp_access_groups ?? []).some((entry: AccessGroupEntry) => accessGroupName(entry) === group);

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

  const isAdmin = userRole !== null && isAdminRole(userRole);
  const {
    tab,
    setTab,
    team: selectedTeam,
    setTeam,
    accessGroup: selectedMcpAccessGroup,
    setAccessGroup,
    search: searchQuery,
    setSearch,
    sort: sortKey,
    setSort,
    selectedServerId,
    selectedServer,
    editServer,
    openServer,
    closeServer,
    envVarsDeepLinkId,
    clearEnvVarsDeepLink,
  } = useMcpServersUrlState(isAdmin, serversWithHealth);
  const [serverIdToDelete, setServerToDelete] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isModalVisible, setModalVisible] = useState(false);
  const [isDiscoveryVisible, setDiscoveryVisible] = useState(false);
  const [isImportVisible, setImportVisible] = useState(false);
  const [prefillData, setPrefillData] = useState<DiscoverableMCPServer | null>(null);
  const [isDeletingServer, setIsDeletingServer] = useState(false);
  const [byokModalServer, setByokModalServer] = useState<MCPServer | null>(null);
  const [envVarsModalServer, setEnvVarsModalServer] = useState<MCPServer | null>(null);
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

  const deepLinkServer = useMemo(
    () => (envVarsDeepLinkId ? serversWithHealth.find((s) => s.server_id === envVarsDeepLinkId) ?? null : null),
    [envVarsDeepLinkId, serversWithHealth],
  );
  const activeEnvVarsServer = envVarsModalServer ?? deepLinkServer;

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
  const teamSelectItems = React.useMemo(
    () => ({
      all: isInternalUser ? "All Available Servers" : "All Servers",
      personal: "Personal",
      ...Object.fromEntries(uniqueTeams.map((team) => [team.team_id, team.team_alias || team.team_id])),
    }),
    [isInternalUser, uniqueTeams],
  );

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

  const accessGroupSelectItems = React.useMemo(
    () => ({
      all: "All Access Groups",
      ...Object.fromEntries(uniqueMcpAccessGroups.map((group) => [group, group])),
    }),
    [uniqueMcpAccessGroups],
  );

  const filteredServers = useMemo(() => {
    if (selectedTeam === "personal") return [];
    return serversWithHealth
      .filter((server) => selectedTeam === "all" || server.teams?.some((team) => team.team_id === selectedTeam))
      .filter((server) => selectedMcpAccessGroup === "all" || inAccessGroup(server, selectedMcpAccessGroup))
      .sort(createdAtDesc);
  }, [serversWithHealth, selectedTeam, selectedMcpAccessGroup]);

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
      toast.success("Deleted MCP Server successfully");
      if (selectedServerId === serverIdToDelete) {
        closeServer();
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

  const handleCreateSuccess = () => {
    setModalVisible(false);
    refetch();
  };

  const handleBack = React.useCallback(() => {
    closeServer();
    refetch();
  }, [closeServer, refetch]);

  if (!accessToken || !userRole || !userID) {
    return <div className="p-6 text-center text-muted-foreground">Missing required authentication parameters.</div>;
  }

  return (
    <TooltipProvider>
      <div className="h-full w-full p-6">
        <AlertDialog open={isDeleteModalOpen} onOpenChange={(open) => !open && cancelDelete()}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete MCP Server?</AlertDialogTitle>
            </AlertDialogHeader>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                This action is permanent and cannot be undone. All associated configurations will be removed.
              </p>

              {serverToDelete && (
                <dl className="mt-3 space-y-1 rounded-lg border border-border bg-muted p-4">
                  {serverToDelete.server_name && (
                    <div className="flex gap-2">
                      <dt className="text-sm text-muted-foreground">Name</dt>
                      <dd className="text-sm font-semibold">{serverToDelete.server_name}</dd>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <dt className="text-sm text-muted-foreground">ID</dt>
                    <dd className="font-mono text-xs">{serverToDelete.server_id}</dd>
                  </div>
                  {serverToDelete.url && (
                    <div className="flex gap-2">
                      <dt className="text-sm text-muted-foreground">URL</dt>
                      <dd className="font-mono text-xs break-all">{serverToDelete.url}</dd>
                    </div>
                  )}
                </dl>
              )}
            </div>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={isDeletingServer}>Cancel</AlertDialogCancel>
              <Button variant="destructive" disabled={isDeletingServer} onClick={confirmDelete}>
                {isDeletingServer ? "Deleting..." : "Delete"}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
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
              <h1 className="text-xl font-semibold">MCP Servers</h1>
              {filteredServers.length > 0 && <Badge variant="secondary">{filteredServers.length}</Badge>}
            </div>
            <p className="mt-1 text-sm text-muted-foreground">Configure and manage your MCP servers</p>
          </div>
          <div className="flex items-center gap-2">
            {isAdminRole(userRole) && (
              <>
                <Button className="shrink-0" variant="secondary" onClick={() => setImportVisible(true)}>
                  Import from JSON
                </Button>
                <Button className="shrink-0" onClick={() => setDiscoveryVisible(true)}>
                  + Add New MCP Server
                </Button>
              </>
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
        <ImportMCPServers
          accessToken={accessToken}
          open={isImportVisible}
          onClose={() => setImportVisible(false)}
          onImported={() => refetch()}
        />
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
        <Tabs value={tab} onValueChange={setTab} className="mt-2 w-full">
          <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0">
            <TabsTrigger value="servers" className="flex-none rounded-none px-4 py-2">
              All Servers
            </TabsTrigger>
            <TabsTrigger value="toolsets" className="flex-none rounded-none px-4 py-2">
              Toolsets
            </TabsTrigger>
            <TabsTrigger value="connect" className="flex-none rounded-none px-4 py-2">
              Connect
            </TabsTrigger>
            {isAdminRole(userRole) && (
              <TabsTrigger value="semantic-filter" className="flex-none rounded-none px-4 py-2">
                Semantic Filter
              </TabsTrigger>
            )}
            {isAdminRole(userRole) && (
              <TabsTrigger value="tool-search" className="flex-none rounded-none px-4 py-2">
                Tool Search
              </TabsTrigger>
            )}
            {isAdminRole(userRole) && (
              <TabsTrigger value="network" className="flex-none rounded-none px-4 py-2">
                Network Settings
              </TabsTrigger>
            )}
            {isAdminRole(userRole) && (
              <TabsTrigger value="submitted" className="flex-none rounded-none px-4 py-2">
                Submitted MCPs
              </TabsTrigger>
            )}
          </TabsList>
          <TabsContent value="servers" keepMounted>
            {selectedServer ? (
              <MCPServerView
                key={selectedServer.server_id}
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
                    <div className="flex items-center gap-6 rounded-lg border border-border bg-card px-4 py-3">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium whitespace-nowrap text-muted-foreground">Team</p>
                        <Select
                          items={teamSelectItems}
                          value={selectedTeam}
                          onValueChange={(v: string | null) => setTeam(v)}
                        >
                          <SelectTrigger className="w-55">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="all">
                              {isInternalUser ? "All Available Servers" : "All Servers"}
                            </SelectItem>
                            <SelectItem value="personal">Personal</SelectItem>
                            {uniqueTeams.map((team) => (
                              <SelectItem key={team.team_id} value={team.team_id}>
                                {team.team_alias || team.team_id}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="h-6 w-px bg-border" />
                      <div className="flex items-center gap-2">
                        <p className="flex items-center text-sm font-medium whitespace-nowrap text-muted-foreground">
                          Access Group
                          <Tooltip>
                            <TooltipTrigger
                              render={
                                <CircleHelp
                                  className="ml-1 size-3.5 text-muted-foreground"
                                  aria-label="About access groups"
                                />
                              }
                            />
                            <TooltipContent>
                              An MCP Access Group is a set of users or teams that have permission to access specific MCP
                              servers. Use access groups to control and organize who can connect to which servers.
                            </TooltipContent>
                          </Tooltip>
                        </p>
                        <Select
                          items={accessGroupSelectItems}
                          value={selectedMcpAccessGroup}
                          onValueChange={(v: string | null) => setAccessGroup(v)}
                        >
                          <SelectTrigger className="w-55">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="all">All Access Groups</SelectItem>
                            {uniqueMcpAccessGroups.map((group) => (
                              <SelectItem key={group} value={group}>
                                {group}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap items-center gap-3">
                  <InputGroup className="max-w-80">
                    <InputGroupAddon>
                      <Search className="size-4 text-muted-foreground" />
                    </InputGroupAddon>
                    <InputGroupInput
                      placeholder="Search by name, alias, URL, or ID"
                      value={searchQuery}
                      onChange={(e) => setSearch(e.target.value)}
                    />
                  </InputGroup>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium whitespace-nowrap text-muted-foreground">Sort</p>
                    <Select items={SORT_OPTIONS} value={sortKey} onValueChange={(v: string | null) => setSort(v)}>
                      <SelectTrigger className="w-55">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {SORT_OPTIONS.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="ml-auto text-xs text-muted-foreground">
                    {displayedServers.length} of {filteredServers.length} servers
                  </div>
                </div>
                <div className="mt-4 w-full">
                  {isLoadingServers ? (
                    <div className="flex items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-card p-12">
                      <UiLoadingSpinner className="size-6 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">Loading MCP servers...</p>
                    </div>
                  ) : displayedServers.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-border bg-card p-12 text-center">
                      <p className="text-sm text-muted-foreground">
                        {filteredServers.length === 0
                          ? "No MCP servers configured. Click '+ Add New MCP Server' to get started."
                          : "No servers match the current filters or search."}
                      </p>
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
                          onClick={() => openServer(server.server_id)}
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
          </TabsContent>
          <TabsContent value="toolsets" keepMounted>
            <MCPToolsetsTab accessToken={accessToken} userRole={userRole} />
          </TabsContent>
          <TabsContent value="connect" keepMounted>
            <MCPConnect />
          </TabsContent>
          {isAdminRole(userRole) && (
            <TabsContent value="semantic-filter" keepMounted>
              <MCPSemanticFilterSettings accessToken={accessToken} />
            </TabsContent>
          )}
          {isAdminRole(userRole) && (
            <TabsContent value="tool-search" keepMounted>
              <MCPToolSearchSettings accessToken={accessToken} />
            </TabsContent>
          )}
          {isAdminRole(userRole) && (
            <TabsContent value="network" keepMounted>
              <MCPNetworkSettings accessToken={accessToken} />
            </TabsContent>
          )}
          {isAdminRole(userRole) && (
            <TabsContent value="submitted" keepMounted>
              <MCPSubmissionsTab accessToken={accessToken} />
            </TabsContent>
          )}
        </Tabs>

        {byokModalServer && (
          <ByokCredentialModal
            server={byokModalServer}
            open={!!byokModalServer}
            onClose={() => setByokModalServer(null)}
            onSuccess={(_serverId) => {
              refetch();
              setByokModalServer(null);
            }}
          />
        )}

        {/* Per-user env-var fill modal — backed by /v1/mcp/server/{id}/user-env-vars */}
        <UserEnvVarsModal
          server={activeEnvVarsServer}
          open={!!activeEnvVarsServer}
          accessToken={accessToken}
          onClose={() => {
            setEnvVarsModalServer(null);
            clearEnvVarsDeepLink();
          }}
          onSaved={() => {
            // Refresh the bulk status so the red "N user fields missing" footer
            // on each card clears once the user has filled in their values.
            refetchEnvVarStatus();
          }}
        />
      </div>
    </TooltipProvider>
  );
};

export default MCPServers;
