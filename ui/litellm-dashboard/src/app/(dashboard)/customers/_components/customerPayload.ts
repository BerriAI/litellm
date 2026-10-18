import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import type { UpdateCustomerRequest } from "@/app/(dashboard)/hooks/customers/customerApi";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import { mcpAllowedToolsFor, mcpServersForIdentifier } from "@/components/mcp_server_management/effectiveMcpServers";
import {
  extractMcpEntitlementUpdate,
  type McpEntitlementUpdate,
} from "@/components/mcp_server_management/mcpEntitlement";

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

const stringsIdentity = (values: readonly string[]): string => JSON.stringify([...new Set(values)].sort());

const optionalCustomerFields = (
  values: CustomerFormValues,
  existing: EndUser | null,
): Partial<UpdateCustomerRequest> => ({
  ...(values.alias.trim() !== (existing?.alias ?? "") ? { alias: values.alias.trim() } : {}),
  ...(values.budget_id && values.budget_id !== existing?.budget_id ? { budget_id: values.budget_id } : {}),
  ...(values.max_budget.trim() && Number(values.max_budget) !== existing?.litellm_budget_table?.max_budget
    ? { max_budget: Number(values.max_budget) }
    : {}),
  ...(values.allowed_model_region && values.allowed_model_region !== existing?.allowed_model_region
    ? { allowed_model_region: values.allowed_model_region }
    : {}),
  ...(values.default_model.trim() && values.default_model.trim() !== existing?.default_model?.trim()
    ? { default_model: values.default_model.trim() }
    : {}),
});

export const buildCustomerPayload = (
  values: CustomerFormValues,
  existing: EndUser | null,
  servers: MCPServer[],
  toolsets: MCPToolset[],
): UpdateCustomerRequest => {
  const grant = extractMcpEntitlementUpdate(values, existing?.object_permission, servers, toolsets);
  return {
    user_id: existing?.user_id ?? values.user_id.trim(),
    ...optionalCustomerFields(values, existing),
    ...(stringsIdentity(values.models) !== stringsIdentity(existing?.models ?? []) ? { models: values.models } : {}),
    ...(grant ? { object_permission: grant } : {}),
  };
};

const customerSecurityIdentity = (customer: EndUser): string => {
  const grant = storedMcpGrant(customer);
  const identity = {
    models: stringsIdentity(customer.models ?? []),
    allowed_model_region: customer.allowed_model_region ?? "",
    default_model: customer.default_model ?? "",
    budget_id: customer.budget_id ?? "",
    max_budget: customer.litellm_budget_table?.max_budget ?? null,
    mcp_servers: stringsIdentity(grant.mcp_servers),
    mcp_access_groups: stringsIdentity(grant.mcp_access_groups),
    mcp_toolsets: stringsIdentity(grant.mcp_toolsets),
    mcp_tool_permissions: Object.keys(grant.mcp_tool_permissions)
      .sort()
      .map((key) => [key, stringsIdentity(grant.mcp_tool_permissions[key])]),
  };
  return JSON.stringify(identity);
};

export const hasCustomerEditConflict = (
  payload: UpdateCustomerRequest,
  baseline: EndUser | null,
  current: EndUser | null,
): boolean => {
  if (!baseline || !current) return baseline !== current;
  if (baseline.user_id !== current.user_id) return true;
  const changesSecurity = Object.keys(payload).some((key) => key !== "user_id" && key !== "alias");
  return changesSecurity && customerSecurityIdentity(baseline) !== customerSecurityIdentity(current);
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
  const grant = extractMcpEntitlementUpdate(values, existing?.object_permission, servers, toolsets);
  if (!grant) return [];
  const unrestricted = unrestrictedDirectServerIds(grant, servers, toolsets);
  const alreadyUnrestricted = unrestrictedDirectServerIds(storedMcpGrant(existing), servers, toolsets);
  return servers.filter((server) => unrestricted.has(server.server_id) && !alreadyUnrestricted.has(server.server_id));
};
