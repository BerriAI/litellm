import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import type { UpdateCustomerRequest } from "@/app/(dashboard)/hooks/customers/customerApi";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import {
  mcpAllowedToolsFor,
  mcpServersForIdentifier,
  mcpToolPermissionKeysFor,
  resolveEffectiveMcpServers,
} from "@/components/mcp_server_management/effectiveMcpServers";
import { extractMcpEntitlement, type McpEntitlementUpdate } from "@/components/mcp_server_management/mcpEntitlement";

export const getCustomerBudgetCounts = (customers: readonly EndUser[]): ReadonlyMap<string, number> =>
  customers.reduce(
    (counts, { budget_id }) => (budget_id ? counts.set(budget_id, (counts.get(budget_id) ?? 0) + 1) : counts),
    new Map<string, number>(),
  );

export const hasSharedCustomerBudget = (customer: EndUser, budgetCounts: ReadonlyMap<string, number>): boolean =>
  Boolean(customer.budget_id && (budgetCounts.get(customer.budget_id) ?? 0) > 1);

export type CustomerFormValues = {
  user_id: string;
  alias: string;
  budget_id: string;
  max_budget: string;
  models: string[];
  allowed_model_region: "" | "eu" | "us";
  default_model: string;
  mcp_servers_and_groups: { servers: string[]; accessGroups: string[]; toolsets: string[] };
  mcp_tool_permissions: Record<string, string[]>;
};

const storedMcpGrant = (customer: EndUser | null): McpEntitlementUpdate => ({
  mcp_servers: customer?.object_permission?.mcp_servers ?? [],
  mcp_access_groups: customer?.object_permission?.mcp_access_groups ?? [],
  mcp_toolsets: customer?.object_permission?.mcp_toolsets ?? [],
  mcp_tool_permissions: customer?.object_permission?.mcp_tool_permissions ?? {},
});

export const toCustomerFormValues = (customer: EndUser | null): CustomerFormValues => {
  const grant = storedMcpGrant(customer);
  const maxBudget = customer?.litellm_budget_table?.max_budget;
  return {
    user_id: customer?.user_id ?? "",
    alias: customer?.alias ?? "",
    budget_id: customer?.budget_id ?? "",
    max_budget: customer?.budget_id || maxBudget == null ? "" : String(maxBudget),
    models: customer?.models ?? [],
    allowed_model_region: customer?.allowed_model_region ?? "",
    default_model: customer?.default_model ?? "",
    mcp_servers_and_groups: {
      servers: grant.mcp_servers,
      accessGroups: grant.mcp_access_groups,
      toolsets: grant.mcp_toolsets,
    },
    mcp_tool_permissions: grant.mcp_tool_permissions,
  };
};

const normalizedStrings = (values: string[]): string[] => Array.from(new Set(values)).toSorted();

const grantIdentity = (grant: McpEntitlementUpdate): string => {
  const normalized = {
    mcp_servers: normalizedStrings(grant.mcp_servers),
    mcp_access_groups: normalizedStrings(grant.mcp_access_groups),
    mcp_toolsets: normalizedStrings(grant.mcp_toolsets),
    mcp_tool_permissions: Object.fromEntries(
      Object.keys(grant.mcp_tool_permissions)
        .toSorted()
        .map((server) => [server, normalizedStrings(grant.mcp_tool_permissions[server])]),
    ),
  };
  return JSON.stringify(normalized);
};

const extractEditedMcpGrant = (
  values: CustomerFormValues,
  storedGrant: McpEntitlementUpdate,
  servers: MCPServer[],
  toolsets: MCPToolset[],
): McpEntitlementUpdate | null => {
  const grant = extractMcpEntitlement(values, servers, toolsets);
  if (!grant) return null;
  const storedSelection = {
    allServers: servers,
    selectedServers: storedGrant.mcp_servers,
    selectedAccessGroups: storedGrant.mcp_access_groups,
    selectedToolsets: storedGrant.mcp_toolsets,
    toolsets,
    toolPermissions: {},
  };
  const selectedServerIds = new Set(resolveEffectiveMcpServers(storedSelection).map(({ server }) => server.server_id));
  // A stored tool-permission key grants every server it names, so servers that only a key reaches survive edits to
  // the selection, including the other servers behind an alias a selected server shares
  const keyOnlyServers = Object.keys(storedGrant.mcp_tool_permissions)
    .flatMap((key) => mcpServersForIdentifier(servers, key))
    .filter((server) => !selectedServerIds.has(server.server_id));
  const keyOnlyServerIds = new Set(keyOnlyServers.map((server) => server.server_id));
  const namesOnlyKeyOnlyServers = (key: string): boolean => {
    const named = mcpServersForIdentifier(servers, key);
    return named.length > 0 && named.every((server) => keyOnlyServerIds.has(server.server_id));
  };
  const independentPermissions = Object.fromEntries(
    Object.entries(values.mcp_tool_permissions).filter(([key]) => namesOnlyKeyOnlyServers(key)),
  );
  const keyOnlyPermissions = Object.fromEntries(
    keyOnlyServers
      .filter((server) => mcpToolPermissionKeysFor(server, independentPermissions, servers).length === 0)
      .flatMap((server) => {
        const allowed = mcpAllowedToolsFor(server, values.mcp_tool_permissions, servers);
        return allowed ? [[server.server_id, [...allowed]]] : [];
      }),
  );
  return {
    ...grant,
    mcp_tool_permissions: { ...grant.mcp_tool_permissions, ...independentPermissions, ...keyOnlyPermissions },
  };
};

const changedMcpGrant = (
  values: CustomerFormValues,
  existing: EndUser | null,
  servers: MCPServer[],
  toolsets: MCPToolset[],
): McpEntitlementUpdate | null => {
  const storedGrant = storedMcpGrant(existing);
  const storedIdentity = grantIdentity(storedGrant);
  const formGrant = {
    mcp_servers: values.mcp_servers_and_groups.servers,
    mcp_access_groups: values.mcp_servers_and_groups.accessGroups,
    mcp_toolsets: values.mcp_servers_and_groups.toolsets,
    mcp_tool_permissions: values.mcp_tool_permissions,
  };
  if (grantIdentity(formGrant) === storedIdentity) return null;
  const grant = extractEditedMcpGrant(values, storedGrant, servers, toolsets);
  return grant && grantIdentity(grant) !== storedIdentity ? grant : null;
};

const optionalCustomerFields = (
  values: CustomerFormValues,
  existing: EndUser | null,
): Partial<UpdateCustomerRequest> => ({
  ...(values.alias.trim() || existing?.alias ? { alias: values.alias.trim() } : {}),
  ...(values.budget_id ? { budget_id: values.budget_id } : {}),
  ...(values.max_budget.trim() ? { max_budget: Number(values.max_budget) } : {}),
  ...(values.allowed_model_region ? { allowed_model_region: values.allowed_model_region } : {}),
  ...(values.default_model.trim() ? { default_model: values.default_model.trim() } : {}),
});

export const buildCustomerPayload = (
  values: CustomerFormValues,
  existing: EndUser | null,
  servers: MCPServer[],
  toolsets: MCPToolset[],
): UpdateCustomerRequest => {
  const grant = changedMcpGrant(values, existing, servers, toolsets);
  return {
    user_id: existing?.user_id ?? values.user_id.trim(),
    ...optionalCustomerFields(values, existing),
    ...(existing || values.models.length > 0 ? { models: values.models } : {}),
    ...(grant ? { object_permission: grant } : {}),
  };
};

const unrestrictedDirectServerIds = (
  grant: McpEntitlementUpdate,
  servers: MCPServer[],
  toolsets: MCPToolset[],
): ReadonlySet<string> => {
  const toolsetServerIds = new Set(
    toolsets
      .filter((toolset) => grant.mcp_toolsets.includes(toolset.toolset_id))
      .flatMap((toolset) => toolset.tools.map((tool) => tool.server_id)),
  );
  return new Set(
    grant.mcp_servers
      .flatMap((identifier) => mcpServersForIdentifier(servers, identifier))
      .filter((server) => !toolsetServerIds.has(server.server_id))
      .filter((server) => mcpAllowedToolsFor(server, grant.mcp_tool_permissions, servers) === undefined)
      .map((server) => server.server_id),
  );
};

export const serversLosingToolLimits = (
  values: CustomerFormValues,
  existing: EndUser | null,
  servers: MCPServer[],
  toolsets: MCPToolset[],
): readonly MCPServer[] => {
  const grant = changedMcpGrant(values, existing, servers, toolsets);
  if (!grant) return [];
  const unrestricted = unrestrictedDirectServerIds(grant, servers, toolsets);
  const alreadyUnrestricted = unrestrictedDirectServerIds(storedMcpGrant(existing), servers, toolsets);
  return servers.filter((server) => unrestricted.has(server.server_id) && !alreadyUnrestricted.has(server.server_id));
};
