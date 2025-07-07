import React, { useState, useEffect } from "react";
import { Text, Badge } from "@tremor/react";
import { ServerIcon } from "@heroicons/react/outline";
import { Tooltip } from "antd";
import { fetchMCPServers } from "../networking";
import { MCPServer } from '../mcp_tools/types';

interface MCPServerPermissionsProps {
  mcpServers: string[];
  accessToken?: string | null;
}

export function MCPServerPermissions({ 
  mcpServers, 
  accessToken 
}: MCPServerPermissionsProps) {
  const [mcpServerDetails, setMCPServerDetails] = useState<MCPServer[]>([]);

  // Fetch MCP server details when component mounts
  useEffect(() => {
    const fetchMCPServerDetails = async () => {
      if (!accessToken || mcpServers.length === 0) return;
      
      try {
        const response = await fetchMCPServers(accessToken);
        if (response && Array.isArray(response)) {
          // Direct array response
          setMCPServerDetails(response);
        } else if (response.data && Array.isArray(response.data)) {
          // Response with data wrapper
          setMCPServerDetails(response.data);
        }
      } catch (error) {
        console.error("Error fetching MCP servers:", error);
      }
    };

    fetchMCPServerDetails();
  }, [accessToken, mcpServers.length]);

  // Function to get display name for MCP server
  const getMCPServerDisplayName = (serverId: string) => {
    const serverDetail = mcpServerDetails.find(server => server.server_id === serverId);
    if (serverDetail) {
      // Truncate server ID: first 3 chars + "..." + last 4 chars
      const truncatedId = serverId.length > 7 
        ? `${serverId.slice(0, 3)}...${serverId.slice(-4)}`
        : serverId;
      return `${serverDetail.alias} (${truncatedId})`;
    }
    return serverId;
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
    </div>
  );
}

export default MCPServerPermissions; 