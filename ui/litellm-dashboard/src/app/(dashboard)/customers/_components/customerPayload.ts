import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import type { UpdateCustomerRequest } from "@/app/(dashboard)/hooks/customers/customerApi";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import { mcpServersForIdentifier } from "@/components/mcp_server_management/effectiveMcpServers";
import { extractMcpEntitlement, type McpEntitlementUpdate } from "@/components/mcp_server_management/mcpEntitlement";

export const getCustomerBudgetCounts = (customers: readonly EndUser[]): ReadonlyMap<string, number> =>
  new Map(
    Array.from(
      Map.groupBy(customers, (customer) => customer.budget_id ?? ""),
      ([budgetId, linkedCustomers]) => [budgetId, linkedCustomers.length] as const,
    ).filter(([budgetId]) => Boolean(budgetId)),
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
  const previousValues = {
    mcp_servers_and_groups: {
      servers: storedGrant.mcp_servers,
      accessGroups: storedGrant.mcp_access_groups,
      toolsets: storedGrant.mcp_toolsets,
    },
    mcp_tool_permissions: storedGrant.mcp_tool_permissions,
  };
  const previousPermissions = extractMcpEntitlement(previousValues, servers, toolsets)?.mcp_tool_permissions ?? {};
  const independentKeys = new Set(
    Object.keys(storedGrant.mcp_tool_permissions).filter((key) => !Object.hasOwn(previousPermissions, key)),
  );
  const independentServerIds = new Set(
    Array.from(independentKeys).flatMap((key) =>
      mcpServersForIdentifier(servers, key).map((server) => server.server_id),
    ),
  );
  const independentPermissions = Object.fromEntries(
    Object.entries(values.mcp_tool_permissions).filter(
      ([key]) =>
        independentKeys.has(key) ||
        mcpServersForIdentifier(servers, key).some((server) => independentServerIds.has(server.server_id)),
    ),
  );
  return { ...grant, mcp_tool_permissions: { ...grant.mcp_tool_permissions, ...independentPermissions } };
};

const changedMcpGrant = (
  values: CustomerFormValues,
  existing: EndUser | null,
  servers: MCPServer[],
  toolsets: MCPToolset[],
): McpEntitlementUpdate | null => {
  const storedGrant = storedMcpGrant(existing);
  const storedIdentity = grantIdentity(storedGrant);
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
    user_id: values.user_id.trim(),
    ...optionalCustomerFields(values, existing),
    ...(existing || values.models.length > 0 ? { models: values.models } : {}),
    ...(grant ? { object_permission: grant } : {}),
  };
};
