import React from "react";
import { Text } from "@tremor/react";
import VectorStorePermissions from "./permissions/VectorStorePermissions";
import MCPServerPermissions from "./permissions/MCPServerPermissions";

interface ObjectPermission {
  object_permission_id: string;
  mcp_servers: string[];
  mcp_access_groups?: string[];
  mcp_tool_permissions?: Record<string, string[]>;
  vector_stores: string[];
}

interface ObjectPermissionsViewProps {
  objectPermission?: ObjectPermission;
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

  const content = (
    <div className={variant === "card" ? "grid grid-cols-1 md:grid-cols-2 gap-6" : "space-y-4"}>
      <VectorStorePermissions vectorStores={vectorStores} accessToken={accessToken} />
      <MCPServerPermissions 
        mcpServers={mcpServers} 
        mcpAccessGroups={mcpAccessGroups} 
        mcpToolPermissions={mcpToolPermissions}
        accessToken={accessToken} 
      />
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
