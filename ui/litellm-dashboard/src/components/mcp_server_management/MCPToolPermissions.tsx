import React, { useEffect, useRef, useState, useMemo } from "react";
import { listMCPTools } from "../networking";
import { MCPTool, MCPServer } from "../mcp_tools/types";
import { Text } from "@tremor/react";
import { Spin, Radio } from "antd";
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
  const fetchToolsForServer = async (serverId: string) => {
    setLoadingTools((prev) => ({ ...prev, [serverId]: true }));
    setToolErrors((prev) => ({ ...prev, [serverId]: "" }));

    try {
      const response = await listMCPTools(accessToken, serverId);

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

  // Auto-fetch tools when servers change
  useEffect(() => {
    servers.forEach((server) => {
      if (!serverTools[server.server_id] && !loadingTools[server.server_id]) {
        fetchToolsForServer(server.server_id);
      }
    });
  }, [servers]);

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
          <div key={server.server_id} className="border rounded-lg bg-gray-50">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-white rounded-t-lg">
              <div>
                <Text className="font-semibold text-gray-900">{serverName}</Text>
                {server.description && <Text className="text-sm text-gray-500">{server.description}</Text>}
              </div>
              <div className="flex items-center gap-3">
                {!disabled && tools.length > 0 && (
                  <Radio.Group
                    value={viewMode}
                    onChange={(e) =>
                      setViewModes((prev) => ({ ...prev, [server.server_id]: e.target.value }))
                    }
                    size="small"
                    optionType="button"
                    buttonStyle="solid"
                    options={[
                      { label: "Risk Groups", value: "crud" },
                      { label: "Flat List", value: "flat" },
                    ]}
                  />
                )}
                {!disabled && (
                  <>
                    <button
                      type="button"
                      className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                      onClick={() => handleSelectAll(server.server_id)}
                      disabled={isLoading}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                      onClick={() => handleDeselectAll(server.server_id)}
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
                  <Spin size="large" />
                  <Text className="ml-3 text-gray-500">Loading tools...</Text>
                </div>
              )}

              {/* Error */}
              {error && !isLoading && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-center">
                  <Text className="text-red-600 font-medium">Unable to load tools</Text>
                  <Text className="text-sm text-red-500 mt-1">{error}</Text>
                </div>
              )}

              {/* CRUD grouped view */}
              {!isLoading && !error && tools.length > 0 && viewMode === "crud" && (
                <McpCrudPermissionPanel
                  tools={tools}
                  value={!toolPermissions[server.server_id] ? undefined : selectedTools}
                  onChange={(allowed) => handleCrudPanelChange(server.server_id, allowed)}
                  readOnly={disabled}
                />
              )}

              {/* Flat list view */}
              {!isLoading && !error && tools.length > 0 && viewMode === "flat" && (
                <div className="space-y-2">
                  {tools.map((tool) => {
                    const isSelected = selectedTools.includes(tool.name);
                    return (
                      <div key={tool.name} className="flex items-start gap-2">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => {
                            if (disabled) return;
                            const next = isSelected
                              ? selectedTools.filter((n) => n !== tool.name)
                              : [...selectedTools, tool.name];
                            handleCrudPanelChange(server.server_id, next);
                          }}
                          disabled={disabled}
                          className="mt-0.5"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <Text className="font-medium text-gray-900">{tool.name}</Text>
                            <Text className="text-sm text-gray-500">
                              - {tool.description || "No description"}
                            </Text>
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
                  <Text className="text-gray-500">No tools available</Text>
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
