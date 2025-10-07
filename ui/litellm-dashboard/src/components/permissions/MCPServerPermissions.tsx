import React, { useState, useEffect } from "react";
import { Text, Badge } from "@tremor/react";
import { ServerIcon } from "@heroicons/react/outline";
import { Tooltip } from "antd";
import { fetchMCPServers } from "../networking";
import { MCPServer } from "../mcp_tools/types";

interface MCPServerPermissionsProps {
  mcpServers: string[];
  mcpAccessGroups?: string[];
  accessToken?: string | null;
}

export function MCPServerPermissions({ mcpServers, mcpAccessGroups = [], accessToken }: MCPServerPermissionsProps) {
  const [mcpServerDetails, setMCPServerDetails] = useState<MCPServer[]>([]);
  const [accessGroupNames, setAccessGroupNames] = useState<string[]>([]);

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

  // Fetch MCP access group names
  useEffect(() => {
    const fetchGroups = async () => {
      if (accessToken && mcpAccessGroups.length > 0) {
        try {
          const groups = await import("../networking").then((m) => m.fetchMCPAccessGroups(accessToken));
          setAccessGroupNames(Array.isArray(groups) ? groups : groups.data || []);
        } catch (error) {
          console.error("Error fetching MCP access groups:", error);
        }
      }
    };
    fetchGroups();
  }, [accessToken, mcpAccessGroups.length]);

  // Function to get display name for MCP server
  const getMCPServerDisplayName = (serverId: string) => {
    const serverDetail = mcpServerDetails.find((server) => server.server_id === serverId);
    if (serverDetail) {
      const truncatedId = serverId.length > 7 ? `${serverId.slice(0, 3)}...${serverId.slice(-4)}` : serverId;
      return `${serverDetail.alias} (${truncatedId})`;
    }
    return serverId;
  };

  // Function to get display name for access group
  const getAccessGroupDisplayName = (group: string) => {
    return group;
  };

  // Merge servers and access groups into one list
  const mergedItems = [
    ...mcpServers.map((server) => ({ type: "server", value: server })),
    ...mcpAccessGroups.map((group) => ({ type: "accessGroup", value: group })),
  ];
  const totalCount = mergedItems.length;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <ServerIcon className="h-4 w-4 text-gray-600" />
        <Text className="font-semibold text-gray-900">MCP Servers</Text>
        <Badge color="gray" size="xs">
          {totalCount}
        </Badge>
      </div>
      {totalCount > 0 ? (
        <div className="flex flex-wrap gap-2">
          {mergedItems.map((item, index) =>
            item.type === "server" ? (
              <Tooltip key={index} title={`Full ID: ${item.value}`} placement="top">
                <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-sm font-medium cursor-help">
                  {getMCPServerDisplayName(item.value)}
                </div>
              </Tooltip>
            ) : (
              <div
                key={index}
                className="inline-flex items-center px-3 py-1.5 rounded-lg bg-green-50 border border-green-200 text-green-800 text-sm font-medium"
              >
                <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                {getAccessGroupDisplayName(item.value)}{" "}
                <span className="ml-1 text-xs text-green-500">(Access Group)</span>
              </div>
            ),
          )}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
          <ServerIcon className="h-4 w-4 text-gray-400" />
          <Text className="text-gray-500 text-sm">No MCP servers or access groups configured</Text>
        </div>
      )}
    </div>
  );
}

export default MCPServerPermissions;
