import { isAdminRole } from "@/utils/roles";
import { HelpCircle as QuestionCircleOutlined } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import NewBadge from "../common_components/NewBadge";
import React, { useEffect, useState, useMemo, useCallback } from "react";
import { useMCPServers } from "../../app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPServerHealth } from "../../app/(dashboard)/hooks/mcpServers/useMCPServerHealth";
import NotificationsManager from "../molecules/notifications_manager";
import { deleteMCPServer } from "../networking";
import { MCPSubmissionsTab } from "./MCPSubmissionsTab";
import { MCPToolsetsTab } from "./MCPToolsetsTab";
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
import { getSecureItem } from "@/utils/secureStorage";

const EDIT_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-edit-state";

const MCPServers: React.FC<MCPServerProps> = ({ accessToken, userRole, userID }) => {
  const { data: mcpServers, isLoading: isLoadingServers, refetch } = useMCPServers();

  const {
    data: healthStatuses,
    isLoading: isLoadingHealth,
    recheckServerHealth,
    recheckingServerIds,
  } = useMCPServerHealth();

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
          server.mcp_access_groups?.some((g: any) =>
            typeof g === "string" ? g === group : g && g.name === group,
          ),
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

  const handleTeamChange = (teamId: string) => {
    setSelectedTeam(teamId);
    filterServers(teamId, selectedMcpAccessGroup);
  };

  const handleMcpAccessGroupChange = (group: string) => {
    setSelectedMcpAccessGroup(group);
    filterServers(selectedTeam, group);
  };

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

  const serverToDelete = serverIdToDelete
    ? (mcpServers || []).find((server) => server.server_id === serverIdToDelete)
    : null;

  const handleCreateSuccess = (newMcpServer: MCPServer) => {
    setFilteredServers((prev) => [...prev, newMcpServer]);
    setModalVisible(false);
    refetch();
  };

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

  const handleBack = React.useCallback(() => {
    setEditServer(false);
    setSelectedServerId(null);
    refetch();
  }, [refetch]);

  if (!accessToken || !userRole || !userID) {
    console.log("Missing required authentication parameters", { accessToken, userRole, userID });
    return (
      <div className="p-6 text-center text-muted-foreground">
        Missing required authentication parameters.
      </div>
    );
  }

  return (
    <div className="w-full h-full p-6">
      <AlertDialog open={isDeleteModalOpen} onOpenChange={(o) => !o && cancelDelete()}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete MCP Server?</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  This action is permanent and cannot be undone. All associated configurations will be
                  removed.
                </p>

                {serverToDelete && (
                  <div className="mt-3 p-4 bg-muted rounded-lg border border-border">
                    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
                      {serverToDelete.server_name && (
                        <>
                          <dt className="text-muted-foreground">Name</dt>
                          <dd className="font-semibold text-foreground">
                            {serverToDelete.server_name}
                          </dd>
                        </>
                      )}
                      <dt className="text-muted-foreground">ID</dt>
                      <dd>
                        <code className="text-xs bg-background px-1.5 py-0.5 rounded">
                          {serverToDelete.server_id}
                        </code>
                      </dd>
                      {serverToDelete.url && (
                        <>
                          <dt className="text-muted-foreground">URL</dt>
                          <dd>
                            <code className="text-xs bg-background px-1.5 py-0.5 rounded break-all">
                              {serverToDelete.url}
                            </code>
                          </dd>
                        </>
                      )}
                    </dl>
                  </div>
                )}
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeletingServer}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                confirmDelete();
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={isDeletingServer}
            >
              {isDeletingServer ? "Deleting..." : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
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
            <h2 className="text-2xl font-semibold m-0">MCP Servers</h2>
            {filteredServers.length > 0 && (
              <span className="inline-flex items-center text-xs font-medium px-2 py-0.5 rounded-full bg-muted text-muted-foreground border border-border">
                {filteredServers.length}
              </span>
            )}
          </div>
          <p className="text-muted-foreground mt-1 text-sm m-0">
            Configure and manage your MCP servers
          </p>
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
      <Tabs defaultValue="servers" className="w-full h-full">
        <TabsList className="mt-2 flex justify-start">
          <TabsTrigger value="servers">All Servers</TabsTrigger>
          <TabsTrigger value="toolsets">Toolsets</TabsTrigger>
          <TabsTrigger value="connect">Connect</TabsTrigger>
          <TabsTrigger value="semantic">Semantic Filter</TabsTrigger>
          <TabsTrigger value="network">Network Settings</TabsTrigger>
          {isAdminRole(userRole) && (
            <TabsTrigger value="submissions">
              <span className="flex items-center gap-2">
                Submitted MCPs <NewBadge />
              </span>
            </TabsTrigger>
          )}
        </TabsList>
        <TabsContent value="servers">
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
                  <div className="flex items-center gap-6 bg-background rounded-lg px-4 py-3 border border-border">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-muted-foreground whitespace-nowrap">
                        Team
                      </span>
                      <Select value={selectedTeam} onValueChange={handleTeamChange}>
                        <SelectTrigger className="w-[220px]" aria-label="Team">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">
                            <span className="font-medium">
                              {isInternalUser ? "All Available Servers" : "All Servers"}
                            </span>
                          </SelectItem>
                          <SelectItem value="personal">
                            <span className="font-medium">Personal</span>
                          </SelectItem>
                          {uniqueTeams.map((team) => (
                            <SelectItem key={team.team_id} value={team.team_id}>
                              <span className="font-medium">{team.team_alias || team.team_id}</span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <Separator orientation="vertical" className="h-6" />
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-muted-foreground whitespace-nowrap flex items-center">
                        Access Group
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <QuestionCircleOutlined className="ml-1 h-3.5 w-3.5 text-muted-foreground" />
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs">
                              An MCP Access Group is a set of users or teams that have permission to
                              access specific MCP servers. Use access groups to control and organize who
                              can connect to which servers.
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </span>
                      <Select
                        value={selectedMcpAccessGroup}
                        onValueChange={handleMcpAccessGroupChange}
                      >
                        <SelectTrigger className="w-[220px]" aria-label="Access Group">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">
                            <span className="font-medium">All Access Groups</span>
                          </SelectItem>
                          {uniqueMcpAccessGroups.map((group) => (
                            <SelectItem key={group} value={group}>
                              <span className="font-medium">{group}</span>
                            </SelectItem>
                          ))}
                        </SelectContent>
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
        </TabsContent>
        <TabsContent value="toolsets">
          <MCPToolsetsTab accessToken={accessToken} userRole={userRole} />
        </TabsContent>
        <TabsContent value="connect">
          <MCPConnect />
        </TabsContent>
        <TabsContent value="semantic">
          <MCPSemanticFilterSettings accessToken={accessToken} />
        </TabsContent>
        <TabsContent value="network">
          <MCPNetworkSettings accessToken={accessToken} />
        </TabsContent>
        {isAdminRole(userRole) && (
          <TabsContent value="submissions">
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
          accessToken={accessToken || ""}
        />
      )}
    </div>
  );
};

export default MCPServers;
