import React, { useState, useEffect } from "react";
import { Text, Badge } from "@tremor/react";
import { ServerIcon } from "@heroicons/react/outline";
import { Tooltip } from "antd";
import { fetchMCPServers } from "../networking";
import { MCPServer } from '../mcp_tools/types';

interface MCPServerPermissionsProps {
  mcpServers: string[];
  mcpAccessGroups?: string[];
  accessToken?: string | null;
}

export function MCPServerPermissions({ 
  mcpServers, 
  mcpAccessGroups = [],
  accessToken 
}: MCPServerPermissionsProps) {
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
          const groups = await import("../networking").then(m => m.fetchMCPAccessGroups(accessToken));
          setAccessGroupNames(Array.isArray(groups) ? groups : (groups.data || []));
        } catch (error) {
          console.error("Error fetching MCP access groups:", error);
        }
      }
    };
    fetchGroups();
  }, [accessToken, mcpAccessGroups.length]);

  // Function to get display name for MCP server
  const getMCPServerDisplayName = (serverId: string) => {
    const serverDetail = mcpServerDetails.find(server => server.server_id === serverId);
    if (serverDetail) {
      const truncatedId = serverId.length > 7 
        ? `${serverId.slice(0, 3)}...${serverId.slice(-4)}`
        : serverId;
      return `${serverDetail.alias} (${truncatedId})`;
    }
    return serverId;
  };

  // Function to get display name for access group
  const getAccessGroupDisplayName = (group: string) => {
    // If accessGroupNames is a list of names, just return the group string
    return group;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <ServerIcon className="h-4 w-4 text-green-600" />
        <Text className="font-semibold text-gray-900">MCP Servers</Text>
        <Badge color="green" size="xs">
          {mcpServers.length}
        </Badge>
      </div>
      {mcpServers.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {mcpServers.map((server, index) => (
            <Tooltip key={index} title={`Full ID: ${server}`} placement="top">
              <div
                className="inline-flex items-center px-3 py-1.5 rounded-lg bg-green-50 border border-green-200 text-green-800 text-sm font-medium cursor-help"
              >
                {getMCPServerDisplayName(server)}
              </div>
            </Tooltip>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
          <ServerIcon className="h-4 w-4 text-gray-400" />
          <Text className="text-gray-500 text-sm">No MCP servers configured</Text>
        </div>
      )}
      <div className="flex items-center gap-2 mt-4">
        <ServerIcon className="h-4 w-4 text-blue-600" />
        <Text className="font-semibold text-gray-900">MCP Access Groups</Text>
        <Badge color="blue" size="xs">
          {mcpAccessGroups.length}
        </Badge>
      </div>
      {mcpAccessGroups.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {mcpAccessGroups.map((group, index) => (
            <div
              key={index}
              className="inline-flex items-center px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-sm font-medium"
            >
              {getAccessGroupDisplayName(group)}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
          <ServerIcon className="h-4 w-4 text-gray-400" />
          <Text className="text-gray-500 text-sm">No MCP access groups configured</Text>
        </div>
      )}
    </div>
  );
}

export default MCPServerPermissions; 