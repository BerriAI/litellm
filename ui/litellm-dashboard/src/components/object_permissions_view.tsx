import React from "react";
import VectorStorePermissions from "./permissions/VectorStorePermissions";
import MCPServerPermissions from "./permissions/MCPServerPermissions";
import AgentPermissions from "./permissions/AgentPermissions";

interface ObjectPermission {
  object_permission_id: string;
  mcp_servers: string[];
  mcp_access_groups?: string[];
  mcp_tool_permissions?: Record<string, string[]>;
  mcp_toolsets?: string[];
  vector_stores: string[];
  agents?: string[];
  agent_access_groups?: string[];
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
  const mcpToolsets = objectPermission?.mcp_toolsets || [];
  const agents = objectPermission?.agents || [];
  const agentAccessGroups = objectPermission?.agent_access_groups || [];

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
      <AgentPermissions
        agents={agents}
        agentAccessGroups={agentAccessGroups}
        accessToken={accessToken}
      />
    </div>
  );

  if (variant === "card") {
    return (
      <div
        className={`bg-background border border-border rounded-lg p-6 ${className}`}
      >
        <div className="flex items-center gap-2 mb-6">
          <div>
            <p className="font-semibold text-foreground">Object Permissions</p>
            <p className="text-xs text-muted-foreground">
              Access control for Vector Stores and MCP Servers
            </p>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <p className="font-medium text-foreground mb-3">Object Permissions</p>
      {content}
    </div>
  );
}

export default ObjectPermissionsView;
