import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { useCreateCustomer, useUpdateCustomer } from "@/app/(dashboard)/hooks/customers/useCustomerMutations";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { buildCustomerPayload, type CustomerFormValues } from "./customerPayload";

export const useSaveCustomer = (existing: EndUser | null) => {
  const { data: servers = [] } = useMCPServers();
  const { data: toolsets = [] } = useMCPToolsets();
  const create = useCreateCustomer();
  const update = useUpdateCustomer();
  const saveCustomer = (values: CustomerFormValues) => {
    const payload = buildCustomerPayload(values, existing, servers, toolsets);
    return existing ? update.mutateAsync(payload) : create.mutateAsync(payload);
  };
  return { saveCustomer, isSaving: create.isPending || update.isPending };
};
