import { allEndUsersCall } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
const customersKeys = createQueryKeys("customers");

export interface Customer {
  user_id: string;
  alias?: string | null;
  spend: number;
  blocked: boolean;
  allowed_model_region?: string | null;
  default_model?: string | null;
  budget_id?: string | null;
  litellm_budget_table?: {
    budget_id: string;
    max_budget?: number | null;
    soft_budget?: number | null;
    max_parallel_requests?: number | null;
    tpm_limit?: number | null;
    rpm_limit?: number | null;
    model_max_budget?: Record<string, unknown> | null;
    budget_duration?: string | null;
    budget_reset_at?: string | null;
    created_at: string;
    created_by: string;
    updated_at: string;
    updated_by: string;
  } | null;
}

export type CustomersResponse = Customer[];

export const useCustomers = () => {
  const { accessToken, userRole } = useAuthorized();
  return useQuery<CustomersResponse>({
    queryKey: customersKeys.list({}),
    queryFn: async () => await allEndUsersCall(accessToken!),
    enabled: Boolean(accessToken) && all_admin_roles.includes(userRole!),
  });
};
