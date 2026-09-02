import React, { useEffect, useRef, useState, useMemo } from "react";
import { listMCPTools } from "../networking";
import { MCPTool } from "../mcp_tools/types";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useMCPServers } from "../../app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "../../app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import McpCrudPermissionPanel from "../mcp_tools/McpCrudPermissionPanel";
import { classifyToolOp } from "../../utils/mcpToolCrudClassification";
import { NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";
import {
  EffectiveMcpServer,
  McpGrantSource,
  applyToolPermissionWrite,
  mcpAllowedToolsFor,
  resolveEffectiveMcpServers,
} from "./effectiveMcpServers";

interface MCPToolPermissionsProps {
  accessToken: string;
  selectedServers: readonly string[];
  selectedAccessGroups?: readonly string[];
  selectedToolsets?: readonly string[];
  toolPermissions: Record<string, string[]>;
  onChange: (toolPermissions: Record<string, string[]>) => void;
  disabled?: boolean;
}

const NO_SELECTION: readonly string[] = [];

interface InheritedBadge {
  readonly label: string;
  readonly className: string;
}

const inheritedBadgeFor = (source: McpGrantSource): InheritedBadge | null => {
  switch (source.kind) {
    case "direct":
      return null;
    case "accessGroup":
      return { label: `Via access group: ${source.name}`, className: "text-green-700 bg-green-50 border-green-200" };
    case "toolset":
      return { label: `Via toolset: ${source.name}`, className: "text-purple-700 bg-purple-50 border-purple-200" };
    case "toolPermission":
      return { label: "Via tool permissions", className: "text-amber-700 bg-amber-50 border-amber-200" };
  }
};

const MCPToolPermissions: React.FC<MCPToolPermissionsProps> = ({
  accessToken,
  selectedServers,
  selectedAccessGroups = NO_SELECTION,
  selectedToolsets = NO_SELECTION,
  toolPermissions,
  onChange,
  disabled = false,
}) => {
  const { data: allServers = [], isError: serversFailed, isLoading: serversLoading } = useMCPServers();
  const { data: toolsets = [], isError: toolsetsFailed } = useMCPToolsets();
  const [serverTools, setServerTools] = useState<Record<string, MCPTool[]>>({});
  const [loadingTools, setLoadingTools] = useState<Record<string, boolean>>({});
  const [toolErrors, setToolErrors] = useState<Record<string, string>>({});
  const [viewModes, setViewModes] = useState<Record<string, "crud" | "flat">>({});

  // Keep a ref to the latest toolPermissions so async fetch callbacks always
  // read the current value and do not overwrite sibling servers' results when
  // multiple fetches complete out-of-order (stale-closure race condition).
  const toolPermissionsRef = useRef(toolPermissions);
  useEffect(() => {
    toolPermissionsRef.current = toolPermissions;
  }, [toolPermissions]);

  // Every server this permission level reaches, not just the directly selected ones: a server
  // reached through an access group or a toolset needs its allowlist visible and editable too.
  const servers = useMemo(
    () =>
      resolveEffectiveMcpServers({
        allServers,
        selectedServers,
        selectedAccessGroups,
        selectedToolsets,
        toolsets,
        toolPermissions,
      }),
    [allServers, selectedServers, selectedAccessGroups, selectedToolsets, toolsets, toolPermissions],
  );

  // Fetch tools for a specific server; applies delete-blocked-by-default for new servers.
  // `token` is passed explicitly so the closure never captures a stale accessToken.
  const fetchToolsForServer = async (entry: EffectiveMcpServer, token: string) => {
    const serverId = entry.server.server_id;
    setLoadingTools((prev) => ({ ...prev, [serverId]: true }));
    setToolErrors((prev) => ({ ...prev, [serverId]: "" }));

    try {
      const response = await listMCPTools(token, serverId);

      if (response.error) {
        setToolErrors((prev) => ({ ...prev, [serverId]: response.message || "Failed to fetch tools" }));
        setServerTools((prev) => ({ ...prev, [serverId]: [] }));
      } else {
        const fetchedTools: MCPTool[] = response.tools || [];
        setServerTools((prev) => ({ ...prev, [serverId]: fetchedTools }));

        // For directly selected servers that have no permissions stored yet, block delete tools by
        // default. Inherited servers are left untouched: writing an entry for one would narrow a
        // grant the admin never edited, just by opening the editor. A server a toolset restricts is
        // left untouched for the opposite reason: the backend unions this entry with the toolset, so
        // the default would widen the grant to every non-delete tool.
        // Read latest permissions from the ref to avoid clobbering concurrent results.
        const latestPermissions = toolPermissionsRef.current;
        const isDirect = entry.source.kind === "direct";
        const unrestricted =
          mcpAllowedToolsFor(entry.server, latestPermissions, allServers) === undefined &&
          entry.toolsetTools === undefined;
        if (isDirect && unrestricted && fetchedTools.length > 0) {
          const nonDeleteTools = fetchedTools
            .filter((t) => classifyToolOp(t.name, t.description || "") !== "delete")
            .map((t) => t.name);
          onChange(applyToolPermissionWrite({ toolPermissions: latestPermissions, entry, allowed: nonDeleteTools }));
        }
      }
    } catch (err) {
      console.error(`Error fetching tools for server ${serverId}:`, err);
      setToolErrors((prev) => ({ ...prev, [serverId]: "Failed to fetch tools" }));
      setServerTools((prev) => ({ ...prev, [serverId]: [] }));
    } finally {
      setLoadingTools((prev) => ({ ...prev, [serverId]: false }));
    }
  };

  // Auto-fetch tools when servers or accessToken change
  useEffect(() => {
    servers.forEach((entry) => {
      const serverId = entry.server.server_id;
      if (!serverTools[serverId] && !loadingTools[serverId]) {
        fetchToolsForServer(entry, accessToken);
      }
    });
    // fetchToolsForServer is defined in this render scope but receives `accessToken`
    // as an explicit argument, so it is safe to omit from deps here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servers, accessToken]);

  // Every write goes through here so an edit is authoritative for the SERVER, not for one of the
  // equivalent keys that may name it.
  const writeAllowedTools = (entry: EffectiveMcpServer, allowed: string[]) => {
    onChange(applyToolPermissionWrite({ toolPermissions, entry, allowed }));
  };

  const handleSelectAll = (entry: EffectiveMcpServer) => {
    const tools = serverTools[entry.server.server_id] || [];
    writeAllowedTools(
      entry,
      tools.map((t) => t.name),
    );
  };

  // The opt-out sentinel short-circuits the backend resolver to zero servers, so nothing stored
  // here is in force and showing a tool matrix would claim otherwise.
  if (selectedServers.includes(NO_MCP_SERVERS_SENTINEL)) {
    return null;
  }

  const selectionSizes = [
    selectedServers.length,
    selectedAccessGroups.length,
    selectedToolsets.length,
    Object.keys(toolPermissions).length,
  ];
  if (!selectionSizes.some((size) => size > 0)) {
    return null;
  }

  return (
    <div className="space-y-4">
      {serversFailed && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800 font-medium">Unable to load MCP servers</p>
          <p className="text-sm text-yellow-700 mt-1">
            This list is incomplete; servers granted directly or through an access group may be missing. Reload before
            changing tool permissions
          </p>
        </div>
      )}

      {toolsetsFailed && selectedToolsets.length > 0 && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800 font-medium">Unable to load toolsets</p>
          <p className="text-sm text-yellow-700 mt-1">
            Servers reached through the selected toolsets are not listed below
          </p>
        </div>
      )}

      {serversLoading && (
        <div className="flex items-center justify-center py-6">
          <UiLoadingSpinner />
          <p className="ml-3 text-sm text-muted-foreground">Loading MCP servers...</p>
        </div>
      )}

      {servers.map((entry) => {
        const server = entry.server;
        const serverId = server.server_id;
        const serverName = server.server_name || server.alias || serverId;
        const tools = serverTools[serverId] || [];
        const selectedTools = entry.allowedTools ?? [];
        const isLoading = loadingTools[serverId];
        const error = toolErrors[serverId];
        const viewMode = viewModes[serverId] ?? "crud";
        const inherited = inheritedBadgeFor(entry.source);
        // The backend adds a toolset's tools to whatever this map allows, so these stay on however
        // the boxes are ticked. Locking them is what keeps the matrix an honest picture of the grant.
        const toolsetTools = entry.toolsetTools ?? [];

        return (
          <div key={serverId} className={`border rounded-lg bg-muted ${inherited ? "border-dashed" : ""}`}>
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-card rounded-t-lg">
              <div>
                <div className="flex items-center gap-2">
                  <p className="text-sm font-semibold text-foreground">{serverName}</p>
                  {inherited && (
                    <span
                      className={`px-1.5 py-0.5 text-[10px] font-semibold border rounded-sm uppercase tracking-wide ${inherited.className}`}
                    >
                      {inherited.label}
                    </span>
                  )}
                </div>
                {server.description && <p className="text-sm text-muted-foreground">{server.description}</p>}
                {entry.ambiguousKeys.length > 0 && (
                  <p className="text-sm text-amber-700 mt-1">
                    {`Also granted by ${entry.ambiguousKeys.map((key) => `"${key}"`).join(", ")}, which names another server too. Those tools stay allowed here until the servers no longer share that name`}
                  </p>
                )}
                {toolsetTools.length > 0 && (
                  <p className="text-sm text-purple-700 mt-1">
                    {toolsetTools.length === 1
                      ? `${toolsetTools[0]} is granted by a selected toolset, so it stays allowed here; edit the toolset to revoke it`
                      : `${toolsetTools.join(", ")} are granted by a selected toolset, so they stay allowed here; edit the toolset to revoke them`}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-3">
                {!disabled && tools.length > 0 && (
                  <RadioGroup
                    value={viewMode}
                    onValueChange={(next) => setViewModes((prev) => ({ ...prev, [serverId]: next as "crud" | "flat" }))}
                    className="flex w-auto items-center gap-4"
                  >
                    <label className="flex items-center gap-2 text-sm">
                      <RadioGroupItem value="crud" />
                      Risk Groups
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                      <RadioGroupItem value="flat" />
                      Flat List
                    </label>
                  </RadioGroup>
                )}
                {!disabled && (
                  <>
                    <button
                      type="button"
                      className="text-sm text-info hover:text-info/80 font-medium"
                      onClick={() => handleSelectAll(entry)}
                      disabled={isLoading}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      className="text-sm text-info hover:text-info/80 font-medium"
                      onClick={() => writeAllowedTools(entry, [])}
                      disabled={isLoading}
                    >
                      Deselect All
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Tools */}
            <div className="p-4">
              {/* Loading */}
              {isLoading && (
                <div className="flex items-center justify-center py-8">
                  <UiLoadingSpinner />
                  <p className="ml-3 text-sm text-muted-foreground">Loading tools...</p>
                </div>
              )}

              {/* Error */}
              {error && !isLoading && (
                <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg text-center">
                  <p className="text-sm text-destructive font-medium">Unable to load tools</p>
                  <p className="text-sm text-destructive mt-1">{error}</p>
                </div>
              )}

              {/* CRUD grouped view */}
              {!isLoading && !error && tools.length > 0 && viewMode === "crud" && (
                <McpCrudPermissionPanel
                  tools={tools}
                  value={entry.allowedTools === undefined ? undefined : [...selectedTools]}
                  lockedTools={toolsetTools}
                  onChange={(allowed) => writeAllowedTools(entry, allowed)}
                  readOnly={disabled}
                />
              )}

              {/* Flat list view */}
              {!isLoading && !error && tools.length > 0 && viewMode === "flat" && (
                <div className="space-y-2">
                  {tools.map((tool) => {
                    const isSelected = selectedTools.includes(tool.name);
                    const isLocked = toolsetTools.includes(tool.name);
                    return (
                      <div key={tool.name} className="flex items-start gap-2">
                        <input
                          type="checkbox"
                          aria-label={tool.name}
                          checked={isSelected}
                          onChange={() => {
                            if (disabled || isLocked) return;
                            const next = isSelected
                              ? selectedTools.filter((n) => n !== tool.name)
                              : [...selectedTools, tool.name];
                            writeAllowedTools(entry, next);
                          }}
                          disabled={disabled || isLocked}
                          className="mt-0.5"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <p className="text-sm font-medium text-foreground">{tool.name}</p>
                            <p className="text-sm text-muted-foreground">- {tool.description || "No description"}</p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Empty State */}
              {!isLoading && !error && tools.length === 0 && (
                <div className="text-center py-6">
                  <p className="text-sm text-muted-foreground">No tools available</p>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default MCPToolPermissions;
