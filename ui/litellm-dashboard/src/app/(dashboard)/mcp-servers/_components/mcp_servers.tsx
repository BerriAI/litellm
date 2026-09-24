import { isAdminRole, isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { CircleHelp, Plug, Search } from "lucide-react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
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
import React, { useEffect, useState, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPServerHealth } from "@/app/(dashboard)/hooks/mcpServers/useMCPServerHealth";
import { toast } from "@/lib/toast";
import { deleteMCPServer } from "@/components/networking";
import { MCPSubmissionsTab } from "./MCPSubmissionsTab";
import { MCPGatewaySessionsTab } from "./MCPGatewaySessionsTab";
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
import { getSecureItem } from "@/utils/secureStorage";
import { TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import { uiHref } from "@/utils/uiHref";
import { cn } from "@/lib/cva.config";
import UserEnvVarsModal from "./UserEnvVarsModal";
import { listMCPUserEnvVarStatus } from "@/components/networking";

export type SortKey = "created_desc" | "updated_desc" | "name_asc" | "health";

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

const compareByName = (a: MCPServer, b: MCPServer): number => {
  const nameA = (a.server_name || a.alias || a.server_id).toLowerCase();
  const nameB = (b.server_name || b.alias || b.server_id).toLowerCase();
  return nameA.localeCompare(nameB) || a.server_id.localeCompare(b.server_id);
};

const compareByTimestampDesc = (a: string | null | undefined, b: string | null | undefined): number => {
  const ta = a ? new Date(a).getTime() : 0;
  const tb = b ? new Date(b).getTime() : 0;
  return tb - ta;
};

export const compareServers = (a: MCPServer, b: MCPServer, sort: SortKey): number => {
  switch (sort) {
    case "name_asc":
      return compareByName(a, b);
    case "updated_desc":
      return compareByTimestampDesc(a.updated_at, b.updated_at) || compareByName(a, b);
    case "health": {
      const ra = HEALTH_RANK[a.status ?? "unknown"] ?? 1;
      const rb = HEALTH_RANK[b.status ?? "unknown"] ?? 1;
      if (ra !== rb) return ra - rb;
      return compareByTimestampDesc(a.created_at, b.created_at) || compareByName(a, b);
    }
    case "created_desc":
    default:
      return compareByTimestampDesc(a.created_at, b.created_at) || compareByName(a, b);
  }
};

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

function DeleteServerDialog({
  open,
  onOpenChange,
  server,
  isDeleting,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  server: MCPServer | undefined;
  isDeleting: boolean;
  onConfirm: () => Promise<void>;
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete MCP Server?</AlertDialogTitle>
        </AlertDialogHeader>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            This action is permanent and cannot be undone. All associated configurations will be removed.
          </p>

          {server && (
            <dl className="mt-3 space-y-1 rounded-lg border border-border bg-muted p-4">
              {server.server_name && (
                <div className="flex gap-2">
                  <dt className="text-sm text-muted-foreground">Name</dt>
                  <dd className="text-sm font-semibold">{server.server_name}</dd>
                </div>
              )}
              <div className="flex gap-2">
                <dt className="text-sm text-muted-foreground">ID</dt>
                <dd className="font-mono text-xs">{server.server_id}</dd>
              </div>
              {server.url && (
                <div className="flex gap-2">
                  <dt className="text-sm text-muted-foreground">URL</dt>
                  <dd className="font-mono text-xs break-all">{server.url}</dd>
                </div>
              )}
            </dl>
          )}
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
          <Button variant="destructive" disabled={isDeleting} onClick={onConfirm}>
            {isDeleting ? "Deleting..." : "Delete"}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

const MCPServers: React.FC<MCPServerProps> = ({ accessToken, userRole, userID, isViewOnly = false }) => {
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

    const healthMap = new Map(healthStatuses.map((h) => [h.server_id, h]));

    return mcpServers.map((server) => {
      const healthStatus = healthMap.get(server.server_id);
      return {
        ...server,
        ...healthStatus,
        status: healthStatus?.status ?? server.status,
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
  const [isImportVisible, setImportVisible] = useState(false);
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

  const serversWithUserFields = useMemo(
    () =>
      new Set((envVarStatuses ?? []).filter((status) => (status.required ?? []).length > 0).map((s) => s.server_id)),
    [envVarStatuses],
  );

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
          server.mcp_access_groups?.some((g: string | { name?: string } | null) =>
            typeof g === "string" ? g === group : g?.name === group,
          ),
        );
      }
      setFilteredServers(filtered);
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
          return [name, alias, url, id].some((value) => value.includes(q));
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
  const serverToDelete = mcpServers?.find((server) => server.server_id === serverIdToDelete);

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
    return <div className="p-6 text-center text-muted-foreground">Missing required authentication parameters.</div>;
  }

  return (
    <TooltipProvider>
      <div className="h-full w-full p-6">
        <DeleteServerDialog
          open={isDeleteModalOpen}
          onOpenChange={(open) => !open && cancelDelete()}
          server={serverToDelete}
          isDeleting={isDeletingServer}
          onConfirm={confirmDelete}
        />
        <CreateMCPServer
          userRole={userRole}
          userID={userID}
          accessToken={accessToken}
          onCreateSuccess={handleCreateSuccess}
          isModalVisible={isModalVisible}
          setModalVisible={setModalVisible}
          availableAccessGroups={uniqueMcpAccessGroups}
          existingServers={mcpServers}
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
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Link href={uiHref("connect")} className={cn(buttonVariants({ variant: "outline" }), "shrink-0")}>
              <Plug />
              My Connections
            </Link>
            {isAdminRole(userRole) ? (
              <>
                <Button className="shrink-0" variant="secondary" onClick={() => setImportVisible(true)}>
                  Import from JSON
                </Button>
                <Button className="shrink-0" onClick={() => setDiscoveryVisible(true)}>
                  + Add New MCP Server
                </Button>
              </>
            ) : (
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
        <Tabs defaultValue="servers" className="mt-2 w-full">
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
              <>
                <TabsTrigger value="semantic-filter" className="flex-none rounded-none px-4 py-2">
                  Semantic Filter
                </TabsTrigger>

                <TabsTrigger value="tool-search" className="flex-none rounded-none px-4 py-2">
                  Tool Search
                </TabsTrigger>

                <TabsTrigger value="network-settings" className="flex-none rounded-none px-4 py-2">
                  Network Settings
                </TabsTrigger>

                <TabsTrigger value="submitted" className="flex-none rounded-none px-4 py-2">
                  Submitted MCPs
                </TabsTrigger>
              </>
            )}
            {isProxyAdminTierRole(userRole) && (
              <TabsTrigger value="connections" className="flex-none rounded-none px-4 py-2">
                Live Connections
              </TabsTrigger>
            )}
          </TabsList>
          <TabsContent value="servers" keepMounted>
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
                isViewOnly={isViewOnly}
                availableAccessGroups={uniqueMcpAccessGroups}
                existingServers={mcpServers}
                initialTabIndex={selectedServerId === toolsTabServerId ? 1 : 0}
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
                          onValueChange={(v: string | null) => handleTeamChange(v ?? "all")}
                        >
                          <SelectTrigger className="w-55" aria-label="Team">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="all">{teamSelectItems.all}</SelectItem>
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
                          onValueChange={(v: string | null) => handleMcpAccessGroupChange(v ?? "all")}
                        >
                          <SelectTrigger className="w-55" aria-label="Access Group">
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
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </InputGroup>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium whitespace-nowrap text-muted-foreground">Sort</p>
                    <Select
                      items={SORT_OPTIONS}
                      value={sortKey}
                      onValueChange={(v: string | null) => setSortKey((v ?? "created_desc") as SortKey)}
                    >
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
                          hasUserFields={serversWithUserFields.has(server.server_id)}
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
          </TabsContent>
          <TabsContent value="toolsets" keepMounted>
            <MCPToolsetsTab accessToken={accessToken} userRole={userRole} />
          </TabsContent>
          <TabsContent value="connect" keepMounted>
            <MCPConnect />
          </TabsContent>
          {isAdminRole(userRole) && (
            <>
              <TabsContent value="semantic-filter" keepMounted>
                <MCPSemanticFilterSettings accessToken={accessToken} />
              </TabsContent>

              <TabsContent value="tool-search" keepMounted>
                <MCPToolSearchSettings accessToken={accessToken} />
              </TabsContent>

              <TabsContent value="network-settings" keepMounted>
                <MCPNetworkSettings accessToken={accessToken} />
              </TabsContent>

              <TabsContent value="submitted" keepMounted>
                <MCPSubmissionsTab accessToken={accessToken} />
              </TabsContent>
            </>
          )}
          {isProxyAdminTierRole(userRole) && (
            <TabsContent value="connections">
              <MCPGatewaySessionsTab
                accessToken={accessToken}
                canTerminate={isProxyAdminRole(userRole) && !isViewOnly}
              />
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
            setDeepLinkServerId(null);
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
