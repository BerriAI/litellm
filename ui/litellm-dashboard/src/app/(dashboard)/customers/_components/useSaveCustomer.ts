import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { useCreateCustomer, useUpdateCustomer } from "@/app/(dashboard)/hooks/customers/useCustomerMutations";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { buildCustomerPayload, type CustomerFormValues } from "./customerPayload";

export const useSaveCustomer = (existing: EndUser | null) => {
  const { data: servers = [], isSuccess: serversReady } = useMCPServers();
  const { data: toolsets = [], isSuccess: toolsetsReady } = useMCPToolsets();
  const create = useCreateCustomer();
  const update = useUpdateCustomer();
  const saveCustomer = (values: CustomerFormValues) => {
    const payload = buildCustomerPayload(values, existing, servers, toolsets);
    const needsToolsets =
      values.mcp_servers_and_groups.toolsets.length > 0 || (existing?.object_permission?.mcp_toolsets?.length ?? 0) > 0;
    const catalogsReady = serversReady && (!needsToolsets || toolsetsReady);
    if (payload.object_permission && !catalogsReady) {
      return Promise.reject(new Error("MCP catalogs must finish loading before permission changes can be saved"));
    }
    return existing ? update.mutateAsync(payload) : create.mutateAsync(payload);
  };
  return { saveCustomer, isSaving: create.isPending || update.isPending };
};
