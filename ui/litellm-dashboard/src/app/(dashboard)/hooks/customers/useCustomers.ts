import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { fetchClient } from "@/lib/http/api";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type { components } from "@/lib/http/schema";

export type EndUser = components["schemas"]["CustomerResponse"];

const customersKeys = createQueryKeys("customers");

export const useCustomers = () => {
  const { accessToken, userRole } = useAuthorized();
  return useQuery({
    queryKey: customersKeys.list({}),
    queryFn: async () => (await fetchClient.GET("/customer/list")).data ?? [],
    enabled: Boolean(accessToken) && all_admin_roles.includes(userRole!),
  });
};
