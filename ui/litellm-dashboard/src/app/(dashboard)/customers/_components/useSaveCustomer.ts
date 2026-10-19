import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { useCreateCustomer, useUpdateCustomer } from "@/app/(dashboard)/hooks/customers/useCustomerMutations";
import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import {
  buildCustomerPayload,
  hasCustomerEditConflict,
  serversLosingToolLimits,
  type CustomerFormValues,
} from "./customerPayload";

export const useSaveCustomer = (existing: EndUser | null) => {
  const { data: servers = [], isSuccess: serversReady } = useMCPServers();
  const { data: toolsets = [], isSuccess: toolsetsReady } = useMCPToolsets();
  const { isSuccess: accessGroupsReady } = useMCPAccessGroups();
  const create = useCreateCustomer();
  const update = useUpdateCustomer();
  const saveCustomer = (values: CustomerFormValues, baseline: EndUser | null = existing) => {
    const payload = buildCustomerPayload(values, baseline, servers, toolsets);
    if (hasCustomerEditConflict(payload, baseline, existing)) {
      return Promise.reject(
        new Error("Customer permissions or budget changed while editing. Cancel and reopen Settings before saving"),
      );
    }
    if (!payload.object_permission) return existing ? update.mutateAsync(payload) : create.mutateAsync(payload);
    const selection = values.mcp_servers_and_groups;
    const stored = baseline?.object_permission;
    const catalogs = [
      { needed: true, ready: serversReady },
      { needed: selection.toolsets.length > 0 || (stored?.mcp_toolsets?.length ?? 0) > 0, ready: toolsetsReady },
      {
        needed: selection.accessGroups.length > 0 || (stored?.mcp_access_groups?.length ?? 0) > 0,
        ready: accessGroupsReady,
      },
    ];
    if (catalogs.some(({ needed, ready }) => needed && !ready)) {
      return Promise.reject(new Error("MCP catalogs must finish loading before permission changes can be saved"));
    }
    if (serversLosingToolLimits(values, baseline, servers, toolsets).length > 0) {
      return Promise.reject(
        new Error(
          "Choose allowed tools for each MCP server before saving, or it gets every tool including delete tools",
        ),
      );
    }
    return existing ? update.mutateAsync(payload) : create.mutateAsync(payload);
  };
  return { saveCustomer, isSaving: create.isPending || update.isPending };
};
