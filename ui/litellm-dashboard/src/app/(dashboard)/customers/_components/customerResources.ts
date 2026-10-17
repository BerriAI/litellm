import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";

export type CustomerResourceCounts = {
  models: number | "all";
  servers: number;
  accessGroups: number;
  toolsets: number;
};

export function getCustomerResourceCounts(
  customer: Pick<EndUser, "models" | "object_permission">,
): CustomerResourceCounts {
  const models = customer.models ?? [];
  const permission = customer.object_permission;
  const servers = permission?.mcp_servers ?? [];
  const modelCount =
    models.length === 0 || models.includes("all-proxy-models") || models.includes("*") ? "all" : new Set(models).size;
  if (servers.includes(NO_MCP_SERVERS_SENTINEL)) {
    return { models: modelCount, servers: 0, accessGroups: 0, toolsets: 0 };
  }
  const serverIdentifiers = [...servers, ...Object.keys(permission?.mcp_tool_permissions ?? {})].filter(
    (server) => server !== ALL_PROXY_MCP_SERVERS_SENTINEL && server !== NO_MCP_SERVERS_SENTINEL,
  );
  return {
    models: modelCount,
    servers: new Set(serverIdentifiers).size,
    accessGroups: new Set(permission?.mcp_access_groups ?? []).size,
    toolsets: new Set(permission?.mcp_toolsets ?? []).size,
  };
}

export type CustomerMcpScope = "limited" | "denied" | "unrestricted";

export function customerMcpScope(permission: EndUser["object_permission"]): CustomerMcpScope {
  const servers = permission?.mcp_servers ?? [];
  if (servers.includes(NO_MCP_SERVERS_SENTINEL)) return "denied";
  const counts = getCustomerResourceCounts({ object_permission: permission });
  if (counts.servers > 0 || counts.accessGroups > 0 || counts.toolsets > 0) return "limited";
  const onlyUnsupportedAll =
    servers.includes(ALL_PROXY_MCP_SERVERS_SENTINEL) ||
    Object.hasOwn(permission?.mcp_tool_permissions ?? {}, ALL_PROXY_MCP_SERVERS_SENTINEL);
  return onlyUnsupportedAll ? "denied" : "unrestricted";
}
