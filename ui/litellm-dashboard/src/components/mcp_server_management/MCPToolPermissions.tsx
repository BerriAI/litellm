import React, { useEffect, useRef, useState, useMemo } from "react";
import { listMCPTools } from "../networking";
import { MCPTool, MCPServer } from "../mcp_tools/types";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group";
import { useMCPServers } from "../../app/(dashboard)/hooks/mcpServers/useMCPServers";
import McpCrudPermissionPanel from "../mcp_tools/McpCrudPermissionPanel";
import { classifyToolOp } from "../../utils/mcpToolCrudClassification";

interface MCPToolPermissionsProps {
  accessToken: string;
  selectedServers: string[];
  toolPermissions: Record<string, string[]>;
  onChange: (toolPermissions: Record<string, string[]>) => void;
  disabled?: boolean;
}

const MCPToolPermissions: React.FC<MCPToolPermissionsProps> = ({
  accessToken,
  selectedServers,
  toolPermissions,
  onChange,
  disabled = false,
}) => {
  const { data: allServers = [] } = useMCPServers();
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

  // Filter servers based on selectedServers
  const servers = useMemo(() => {
    if (selectedServers.length === 0) return [];
    return allServers.filter((server: MCPServer) => selectedServers.includes(server.server_id));
  }, [allServers, selectedServers]);

  // Fetch tools for a specific server; applies delete-blocked-by-default for new servers.
  // `token` is passed explicitly so the closure never captures a stale accessToken.
  const fetchToolsForServer = async (serverId: string, token: string) => {
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

        // For servers that have no permissions stored yet, block delete tools by default.
        // Read latest permissions from the ref to avoid clobbering concurrent results.
        const latestPermissions = toolPermissionsRef.current;
        if (!latestPermissions[serverId] && fetchedTools.length > 0) {
          const nonDeleteTools = fetchedTools
            .filter((t) => classifyToolOp(t.name, t.description || "") !== "delete")
            .map((t) => t.name);
          onChange({ ...latestPermissions, [serverId]: nonDeleteTools });
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
    servers.forEach((server) => {
      if (!serverTools[server.server_id] && !loadingTools[server.server_id]) {
        fetchToolsForServer(server.server_id, accessToken);
      }
    });
    // fetchToolsForServer is defined in this render scope but receives `accessToken`
    // as an explicit argument, so it is safe to omit from deps here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servers, accessToken]);

  const handleCrudPanelChange = (serverId: string, allowed: string[]) => {
    onChange({ ...toolPermissions, [serverId]: allowed });
  };

  const handleSelectAll = (serverId: string) => {
    const tools = serverTools[serverId] || [];
    onChange({ ...toolPermissions, [serverId]: tools.map((t) => t.name) });
  };

  const handleDeselectAll = (serverId: string) => {
    onChange({ ...toolPermissions, [serverId]: [] });
  };

  if (selectedServers.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      {servers.map((server) => {
        const serverName = server.server_name || server.alias || server.server_id;
        const tools = serverTools[server.server_id] || [];
        const selectedTools = toolPermissions[server.server_id] || [];
        const isLoading = loadingTools[server.server_id];
        const error = toolErrors[server.server_id];
        const viewMode = viewModes[server.server_id] ?? "crud";

        return (
          <div
            key={server.server_id}
            className="border border-border rounded-lg bg-muted"
          >
            <div className="flex items-center justify-between p-4 border-b border-border bg-background rounded-t-lg">
              <div>
                <p className="font-semibold text-foreground">{serverName}</p>
                {server.description && (
                  <p className="text-sm text-muted-foreground">
                    {server.description}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-3">
                {!disabled && tools.length > 0 && (
                  <ToggleGroup
                    type="single"
                    size="sm"
                    value={viewMode}
                    onValueChange={(v) => {
                      if (!v) return;
                      setViewModes((prev) => ({
                        ...prev,
                        [server.server_id]: v as "crud" | "flat",
                      }));
                    }}
                  >
                    <ToggleGroupItem value="crud">Risk Groups</ToggleGroupItem>
                    <ToggleGroupItem value="flat">Flat List</ToggleGroupItem>
                  </ToggleGroup>
                )}
                {!disabled && (
                  <>
                    <button
                      type="button"
                      className="text-sm text-primary hover:underline font-medium"
                      onClick={() => handleSelectAll(server.server_id)}
                      disabled={isLoading}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      className="text-sm text-primary hover:underline font-medium"
                      onClick={() => handleDeselectAll(server.server_id)}
                      disabled={isLoading}
                    >
                      Deselect All
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className="p-4">
              {isLoading && (
                <div className="flex items-center justify-center py-8 gap-3">
                  <Skeleton className="h-8 w-8 rounded-full" />
                  <p className="text-muted-foreground">Loading tools...</p>
                </div>
              )}

              {error && !isLoading && (
                <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg text-center">
                  <p className="text-destructive font-medium">
                    Unable to load tools
                  </p>
                  <p className="text-sm text-destructive mt-1">{error}</p>
                </div>
              )}

              {!isLoading && !error && tools.length > 0 && viewMode === "crud" && (
                <McpCrudPermissionPanel
                  tools={tools}
                  value={
                    !toolPermissions[server.server_id] ? undefined : selectedTools
                  }
                  onChange={(allowed) =>
                    handleCrudPanelChange(server.server_id, allowed)
                  }
                  readOnly={disabled}
                />
              )}

              {!isLoading && !error && tools.length > 0 && viewMode === "flat" && (
                <div className="space-y-2">
                  {tools.map((tool) => {
                    const isSelected = selectedTools.includes(tool.name);
                    return (
                      <div key={tool.name} className="flex items-start gap-2">
                        <Checkbox
                          checked={isSelected}
                          disabled={disabled}
                          className="mt-0.5"
                          onCheckedChange={(next) => {
                            if (disabled) return;
                            const nextList = next
                              ? [...selectedTools, tool.name]
                              : selectedTools.filter((n) => n !== tool.name);
                            handleCrudPanelChange(server.server_id, nextList);
                          }}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground">
                              {tool.name}
                            </span>
                            <span className="text-sm text-muted-foreground">
                              - {tool.description || "No description"}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {!isLoading && !error && tools.length === 0 && (
                <div className="text-center py-6">
                  <p className="text-muted-foreground">No tools available</p>
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
