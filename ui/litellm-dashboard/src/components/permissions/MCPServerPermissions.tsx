import React, { useState, useEffect } from "react";
import { Text, Badge } from "@tremor/react";
import { ServerIcon, ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { Tooltip } from "antd";
import { fetchMCPServers, fetchMCPToolsets } from "../networking";
import { MCPServer, MCPToolset } from "../mcp_tools/types";

interface MCPServerPermissionsProps {
  mcpServers: string[];
  mcpAccessGroups?: string[];
  mcpToolPermissions?: Record<string, string[]>;
  mcpToolsets?: string[];
  accessToken?: string | null;
}

export function MCPServerPermissions({
  mcpServers,
  mcpAccessGroups = [],
  mcpToolPermissions = {},
  mcpToolsets = [],
  accessToken
}: MCPServerPermissionsProps) {
  const [mcpServerDetails, setMCPServerDetails] = useState<MCPServer[]>([]);
  const [toolsetDetails, setToolsetDetails] = useState<MCPToolset[]>([]);
  const [expandedServers, setExpandedServers] = useState<Set<string>>(new Set());
  const [expandedToolsets, setExpandedToolsets] = useState<Set<string>>(new Set());

  const toggleServerExpansion = (serverId: string) => {
    setExpandedServers((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(serverId)) {
        newSet.delete(serverId);
      } else {
        newSet.add(serverId);
      }
      return newSet;
    });
  };

  const toggleToolsetExpansion = (toolsetId: string) => {
    setExpandedToolsets((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(toolsetId)) {
        newSet.delete(toolsetId);
      } else {
        newSet.add(toolsetId);
      }
      return newSet;
    });
  };

  // Fetch MCP server details when component mounts
  useEffect(() => {
    const fetchMCPServerDetails = async () => {
      if (accessToken && mcpServers.length > 0) {
        try {
          const response = await fetchMCPServers(accessToken);
          if (response && Array.isArray(response)) {
            setMCPServerDetails(response);
          } else if (response.data && Array.isArray(response.data)) {
            setMCPServerDetails(response.data);
          }
        } catch (error) {
          console.error("Error fetching MCP servers:", error);
        }
      }
    };
    fetchMCPServerDetails();
  }, [accessToken, mcpServers.length]);

  // Fetch toolset details
  useEffect(() => {
    const fetchToolsets = async () => {
      if (accessToken && mcpToolsets.length > 0) {
        try {
          const all = await fetchMCPToolsets(accessToken);
          const filtered = Array.isArray(all)
            ? all.filter((t: MCPToolset) => mcpToolsets.includes(t.toolset_id))
            : [];
          setToolsetDetails(filtered);
        } catch (error) {
          console.error("Error fetching toolsets:", error);
        }
      }
    };
    fetchToolsets();
  }, [accessToken, mcpToolsets.length]);

  // Function to get display name for MCP server
  const getMCPServerDisplayName = (serverId: string) => {
    const serverDetail = mcpServerDetails.find((server) => server.server_id === serverId);
    if (serverDetail) {
      const truncatedId = serverId.length > 7 ? `${serverId.slice(0, 3)}...${serverId.slice(-4)}` : serverId;
      return `${serverDetail.alias} (${truncatedId})`;
    }
    return serverId;
  };

  // Merge servers and access groups into one list
  const mergedItems = [
    ...mcpServers.map((server) => ({ type: "server", value: server })),
    ...mcpAccessGroups.map((group) => ({ type: "accessGroup", value: group })),
  ];
  const totalCount = mergedItems.length + mcpToolsets.length;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <ServerIcon className="h-4 w-4 text-blue-600" />
        <Text className="font-semibold text-gray-900">MCP Servers</Text>
        <Badge color="blue" size="xs">
          {totalCount}
        </Badge>
      </div>

      {totalCount > 0 ? (
        <div className="max-h-[400px] overflow-y-auto space-y-2 pr-1">
          {mergedItems.map((item, index) => {
            const toolsForServer = item.type === "server" ? mcpToolPermissions[item.value] : undefined;
            const hasToolRestrictions = toolsForServer && toolsForServer.length > 0;
            const isExpanded = expandedServers.has(item.value);

            return (
              <div key={index} className="space-y-2">
                <div
                  onClick={() => hasToolRestrictions && toggleServerExpansion(item.value)}
                  className={`flex items-center gap-3 py-2 px-3 rounded-lg border border-gray-200 transition-all ${
                    hasToolRestrictions
                      ? 'cursor-pointer hover:bg-gray-50 hover:border-gray-300'
                      : 'bg-white'
                  }`}
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    {item.type === "server" ? (
                      <Tooltip title={`Full ID: ${item.value}`} placement="top">
                        <div className="inline-flex items-center gap-2 min-w-0">
                          <span className="inline-block w-1.5 h-1.5 bg-blue-500 rounded-full flex-shrink-0"></span>
                          <span className="text-sm font-medium text-gray-900 truncate">{getMCPServerDisplayName(item.value)}</span>
                        </div>
                      </Tooltip>
                    ) : (
                      <div className="inline-flex items-center gap-2 min-w-0">
                        <span className="inline-block w-1.5 h-1.5 bg-green-500 rounded-full flex-shrink-0"></span>
                        <span className="text-sm font-medium text-gray-900 truncate">{item.value}</span>
                        <span className="ml-1 px-1.5 py-0.5 text-[9px] font-semibold text-green-600 bg-green-50 border border-green-200 rounded uppercase tracking-wide flex-shrink-0">
                          Group
                        </span>
                      </div>
                    )}
                  </div>

                  {hasToolRestrictions && (
                    <div className="flex items-center gap-1 flex-shrink-0 whitespace-nowrap">
                      <span className="text-xs font-medium text-gray-600">{toolsForServer.length}</span>
                      <span className="text-xs text-gray-500">{toolsForServer.length === 1 ? "tool" : "tools"}</span>
                      {isExpanded ? (
                        <ChevronDownIcon className="h-3.5 w-3.5 text-gray-400 ml-0.5" />
                      ) : (
                        <ChevronRightIcon className="h-3.5 w-3.5 text-gray-400 ml-0.5" />
                      )}
                    </div>
                  )}
                </div>

                {/* Show tool permissions if expanded */}
                {hasToolRestrictions && isExpanded && (
                  <div className="ml-4 pl-4 border-l-2 border-blue-200 pb-1">
                    <div className="flex flex-wrap gap-1.5">
                      {toolsForServer.map((tool, toolIndex) => (
                        <span
                          key={toolIndex}
                          className="inline-flex items-center px-2.5 py-1 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-xs font-medium"
                        >
                          {tool}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Toolsets section */}
          {mcpToolsets.length > 0 && mcpToolsets.map((toolsetId, index) => {
            const detail = toolsetDetails.find((t) => t.toolset_id === toolsetId);
            const isExpanded = expandedToolsets.has(toolsetId);
            const toolCount = detail?.tools.length ?? 0;

            return (
              <div key={`toolset-${index}`} className="space-y-2">
                <div
                  onClick={() => toolCount > 0 && toggleToolsetExpansion(toolsetId)}
                  className={`flex items-center gap-3 py-2 px-3 rounded-lg border border-purple-200 transition-all ${
                    toolCount > 0 ? 'cursor-pointer hover:bg-purple-50 hover:border-purple-300' : 'bg-white'
                  }`}
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="inline-block w-1.5 h-1.5 bg-purple-500 rounded-full flex-shrink-0"></span>
                    <span className="text-sm font-medium text-gray-900 truncate">
                      {detail?.toolset_name ?? toolsetId}
                    </span>
                    <span className="ml-1 px-1.5 py-0.5 text-[9px] font-semibold text-purple-600 bg-purple-50 border border-purple-200 rounded uppercase tracking-wide flex-shrink-0">
                      Toolset
                    </span>
                  </div>
                  {toolCount > 0 && (
                    <div className="flex items-center gap-1 flex-shrink-0 whitespace-nowrap">
                      <span className="text-xs font-medium text-gray-600">{toolCount}</span>
                      <span className="text-xs text-gray-500">{toolCount === 1 ? "tool" : "tools"}</span>
                      {isExpanded ? (
                        <ChevronDownIcon className="h-3.5 w-3.5 text-gray-400 ml-0.5" />
                      ) : (
                        <ChevronRightIcon className="h-3.5 w-3.5 text-gray-400 ml-0.5" />
                      )}
                    </div>
                  )}
                </div>

                {toolCount > 0 && isExpanded && detail && (
                  <div className="ml-4 pl-4 border-l-2 border-purple-200 pb-1">
                    <div className="flex flex-wrap gap-1.5">
                      {detail.tools.map((tool, toolIndex) => (
                        <span
                          key={toolIndex}
                          className="inline-flex items-center px-2.5 py-1 rounded-lg bg-purple-50 border border-purple-200 text-purple-800 text-xs font-medium"
                        >
                          <span className="text-purple-400 mr-1 text-[10px]">{tool.server_id.slice(0, 6)}…</span>
                          {tool.tool_name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
          <ServerIcon className="h-4 w-4 text-gray-400" />
          <Text className="text-gray-500 text-sm">No MCP servers, access groups, or toolsets configured</Text>
        </div>
      )}
    </div>
  );
}

export default MCPServerPermissions;
