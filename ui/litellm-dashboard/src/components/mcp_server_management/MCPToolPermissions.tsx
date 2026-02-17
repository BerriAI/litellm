import React, { useEffect, useState, useMemo } from "react";
import { listMCPTools } from "../networking";
import { MCPTool, MCPServer } from "../mcp_tools/types";
import { Text } from "@tremor/react";
import { Spin, Checkbox } from "antd";
import { XIcon } from "lucide-react";
import { useMCPServers } from "../../app/(dashboard)/hooks/mcpServers/useMCPServers";

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

  // Filter servers based on selectedServers
  const servers = useMemo(() => {
    if (selectedServers.length === 0) return [];
    return allServers.filter((server: MCPServer) => selectedServers.includes(server.server_id));
  }, [allServers, selectedServers]);

  // Fetch tools for a specific server
  const fetchToolsForServer = async (serverId: string) => {
    setLoadingTools((prev) => ({ ...prev, [serverId]: true }));
    setToolErrors((prev) => ({ ...prev, [serverId]: "" }));

    try {
      const response = await listMCPTools(accessToken, serverId);

      if (response.error) {
        setToolErrors((prev) => ({ ...prev, [serverId]: response.message || "Failed to fetch tools" }));
        setServerTools((prev) => ({ ...prev, [serverId]: [] }));
      } else {
        setServerTools((prev) => ({ ...prev, [serverId]: response.tools || [] }));
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

  // Handle tool selection
  const handleToolToggle = (serverId: string, toolName: string) => {
    const currentTools = toolPermissions[serverId] || [];
    const newTools = currentTools.includes(toolName)
      ? currentTools.filter((name) => name !== toolName)
      : [...currentTools, toolName];

    const updatedPermissions = {
      ...toolPermissions,
      [serverId]: newTools,
    };
    onChange(updatedPermissions);
  };

  const handleSelectAll = (serverId: string) => {
    const tools = serverTools[serverId] || [];
    const newPermissions = {
      ...toolPermissions,
      [serverId]: tools.map((t) => t.name),
    };
    onChange(newPermissions);
  };

  const handleDeselectAll = (serverId: string) => {
    const newPermissions = {
      ...toolPermissions,
      [serverId]: [],
    };
    onChange(newPermissions);
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

        return (
          <div key={server.server_id} className="border rounded-lg bg-gray-50">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-white rounded-t-lg">
              <div>
                <Text className="font-semibold text-gray-900">{serverName}</Text>
                {server.description && <Text className="text-sm text-gray-500">{server.description}</Text>}
              </div>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                  onClick={() => handleSelectAll(server.server_id)}
                  disabled={disabled || isLoading}
                >
                  Select All
                </button>
                <button
                  type="button"
                  className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                  onClick={() => handleDeselectAll(server.server_id)}
                  disabled={disabled || isLoading}
                >
                  Deselect All
                </button>
                <button
                  type="button"
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => {
                    // Handle remove server if needed
                  }}
                >
                  <XIcon className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Tools */}
            <div className="p-4">
              <Text className="text-sm font-medium text-gray-700 mb-3">Available Tools</Text>

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

              {/* Tool List - Compact */}
              {!isLoading && !error && tools.length > 0 && (
                <div className="space-y-2">
                  {tools.map((tool) => {
                    const isSelected = selectedTools.includes(tool.name);

                    return (
                      <div key={tool.name} className="flex items-start gap-2">
                        <Checkbox
                          checked={isSelected}
                          onChange={() => handleToolToggle(server.server_id, tool.name)}
                          disabled={disabled}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <Text className="font-medium text-gray-900">{tool.name}</Text>
                            <Text className="text-sm text-gray-500">- {tool.description || "No description"}</Text>
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
