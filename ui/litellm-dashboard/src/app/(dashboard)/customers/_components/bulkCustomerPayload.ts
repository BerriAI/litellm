import type { BulkUpdateCustomersRequest, UpdateCustomerRequest } from "@/app/(dashboard)/hooks/customers/customerApi";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";

import type { MCPServer } from "@/components/mcp_tools/types";
import { mcpAllowedToolsFor, mcpServersForIdentifier } from "@/components/mcp_server_management/effectiveMcpServers";

import { hasSharedCustomerBudget } from "./customerPayload";

export const MAX_BULK_CUSTOMERS = 500;

type OptIn<Value> = { enabled: boolean; value: Value };

export type BulkCustomerEdits = {
  budget: { enabled: boolean; mode: "link" | "max"; budgetId: string; maxBudget: string };
  models: OptIn<string[]>;
  region: OptIn<"" | "eu" | "us">;
  defaultModel: OptIn<string>;
  mcpServers: OptIn<string[]>;
  mcpAccessGroups: OptIn<string[]>;
  mcpToolsets: OptIn<string[]>;
};

export const emptyBulkCustomerEdits = (): BulkCustomerEdits => ({
  budget: { enabled: false, mode: "link", budgetId: "", maxBudget: "" },
  models: { enabled: false, value: [] },
  region: { enabled: false, value: "" },
  defaultModel: { enabled: false, value: "" },
  mcpServers: { enabled: false, value: [] },
  mcpAccessGroups: { enabled: false, value: [] },
  mcpToolsets: { enabled: false, value: [] },
});

type BulkCustomerPayloadResult = { ok: true; payload: BulkUpdateCustomersRequest } | { ok: false; error: string };

const validateBulkCustomerEdits = (
  customers: EndUser[],
  edits: BulkCustomerEdits,
  allCustomers: readonly EndUser[],
): string | null => {
  if (customers.length === 0) return "Select at least one customer";
  if (customers.length > MAX_BULK_CUSTOMERS) {
    return `Select at most ${MAX_BULK_CUSTOMERS} customers`;
  }
  if (!Object.values(edits).some((field) => field.enabled)) {
    return "Choose at least one field to update";
  }
  if (edits.budget.enabled && edits.budget.mode === "link" && !edits.budget.budgetId.trim()) {
    return "Select an existing budget";
  }
  if (edits.budget.enabled && edits.budget.mode === "max") {
    if (customers.some((customer) => hasSharedCustomerBudget(customer, allCustomers))) {
      return "Max Budget cannot be changed for customers with shared budgets. Select an existing budget instead";
    }
    if (!Number.isFinite(Number(edits.budget.maxBudget)) || Number(edits.budget.maxBudget) <= 0) {
      return "Enter a maximum budget greater than zero";
    }
  }
  if (edits.region.enabled && !["eu", "us"].includes(edits.region.value)) {
    return "Choose EU or US for the allowed region";
  }
  if (edits.defaultModel.enabled && !edits.defaultModel.value.trim()) {
    return "Select a default model";
  }

  return null;
};

const retainedToolPermissions = (
  customer: EndUser,
  selectedServers: readonly string[],
  servers: readonly MCPServer[],
): Record<string, string[]> => {
  const permissions = customer.object_permission?.mcp_tool_permissions ?? {};
  const selectedIds = new Set(
    selectedServers.flatMap((identifier) =>
      mcpServersForIdentifier(servers, identifier).map((server) => server.server_id),
    ),
  );
  const knownEntries = servers
    .filter((server) => selectedIds.has(server.server_id))
    .flatMap((server) => {
      const tools = mcpAllowedToolsFor(server, permissions, servers);
      return tools === undefined ? [] : [[server.server_id, [...tools]]];
    });
  const unresolvedEntries = Object.entries(permissions).filter(
    ([identifier]) => selectedServers.includes(identifier) && mcpServersForIdentifier(servers, identifier).length === 0,
  );
  return Object.fromEntries([...knownEntries, ...unresolvedEntries]);
};

export const buildBulkCustomerPayload = (
  customers: EndUser[],
  edits: BulkCustomerEdits,
  servers: readonly MCPServer[],
  allCustomers: readonly EndUser[],
): BulkCustomerPayloadResult => {
  const error = validateBulkCustomerEdits(customers, edits, allCustomers);
  if (error) return { ok: false, error };

  const objectPermission = {
    ...(edits.mcpServers.enabled ? { mcp_servers: edits.mcpServers.value } : {}),
    ...(edits.mcpAccessGroups.enabled ? { mcp_access_groups: edits.mcpAccessGroups.value } : {}),
    ...(edits.mcpToolsets.enabled ? { mcp_toolsets: edits.mcpToolsets.value } : {}),
  };
  const updates: Omit<UpdateCustomerRequest, "user_id"> = {
    ...(edits.budget.enabled && edits.budget.mode === "link" ? { budget_id: edits.budget.budgetId.trim() } : {}),
    ...(edits.budget.enabled && edits.budget.mode === "max" ? { max_budget: Number(edits.budget.maxBudget) } : {}),
    ...(edits.models.enabled ? { models: edits.models.value } : {}),
    ...(edits.region.enabled && edits.region.value ? { allowed_model_region: edits.region.value } : {}),
    ...(edits.defaultModel.enabled ? { default_model: edits.defaultModel.value.trim() } : {}),
  };
  return {
    ok: true,
    payload: {
      customers: customers.map((customer) => ({
        user_id: customer.user_id,
        ...updates,
        ...(Object.keys(objectPermission).length > 0
          ? {
              object_permission: {
                ...objectPermission,
                ...(edits.mcpServers.enabled
                  ? { mcp_tool_permissions: retainedToolPermissions(customer, edits.mcpServers.value, servers) }
                  : {}),
              },
            }
          : {}),
      })),
    },
  };
};
