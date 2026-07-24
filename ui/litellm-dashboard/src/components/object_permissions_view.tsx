import React from "react";
import { Text } from "@tremor/react";
import VectorStorePermissions from "./permissions/VectorStorePermissions";
import MCPServerPermissions from "./permissions/MCPServerPermissions";
import AgentPermissions from "./permissions/AgentPermissions";
import type { ObjectPermission } from "./object_permission_types";

interface ObjectPermissionsViewProps {
  objectPermission?: ObjectPermission | null;
  variant?: "card" | "inline";
  className?: string;
  accessToken?: string | null;
}

export function ObjectPermissionsView({
  objectPermission,
  variant = "card",
  className = "",
  accessToken,
}: ObjectPermissionsViewProps) {
  const vectorStores = objectPermission?.vector_stores || [];
  const mcpServers = objectPermission?.mcp_servers || [];
  const mcpAccessGroups = objectPermission?.mcp_access_groups || [];
  const mcpToolPermissions = objectPermission?.mcp_tool_permissions || {};
  const mcpToolsets = objectPermission?.mcp_toolsets || [];
  const agents = objectPermission?.agents || [];
  const agentAccessGroups = objectPermission?.agent_access_groups || [];
  const searchTools = objectPermission?.search_tools || [];

  const content = (
    <div className={variant === "card" ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" : "space-y-4"}>
      <VectorStorePermissions vectorStores={vectorStores} accessToken={accessToken} />
      <MCPServerPermissions
        mcpServers={mcpServers}
        mcpAccessGroups={mcpAccessGroups}
        mcpToolPermissions={mcpToolPermissions}
        mcpToolsets={mcpToolsets}
        accessToken={accessToken}
      />
      <AgentPermissions agents={agents} agentAccessGroups={agentAccessGroups} accessToken={accessToken} />
      <div className="rounded-md border border-gray-100 p-4">
        <Text className="text-sm font-medium text-gray-800">Search tools</Text>
        {searchTools.length === 0 ? (
          <Text className="mt-1 block text-xs text-gray-500">
            No restriction — all configured search tools are allowed for this team.
          </Text>
        ) : (
          <Text className="mt-1 block text-xs text-gray-700">{searchTools.join(", ")}</Text>
        )}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`bg-white border border-gray-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-6">
          <div>
            <Text className="font-semibold text-gray-900">Object Permissions</Text>
            <Text className="text-xs text-gray-500">Access control for Vector Stores and MCP Servers</Text>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <Text className="font-medium text-gray-900 mb-3">Object Permissions</Text>
      {content}
    </div>
  );
}

export default ObjectPermissionsView;
