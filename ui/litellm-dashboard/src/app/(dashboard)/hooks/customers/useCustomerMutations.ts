import { useMutation, useQueryClient } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import { all_admin_roles } from "@/utils/roles";
import { createCustomer, deleteCustomers, setCustomerBlockedState, updateCustomer } from "./customerApi";

export const useCustomer = (id: string | null) => {
  const { accessToken, userRole } = useAuthorized();
  return $api.useQuery(
    "get",
    "/customer/info",
    { params: { query: { end_user_id: id ?? "" } } },
    { enabled: Boolean(accessToken && id) && all_admin_roles.includes(userRole ?? "") },
  );
};

const useCustomerMutation = <TInput, TResult>(action: (accessToken: string, input: TInput) => Promise<TResult>) => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  return useMutation<TResult, Error, TInput>({
    mutationFn: (input) => {
      if (!accessToken) throw new Error("Access token is required");
      return action(accessToken, input);
    },
    onSuccess: () =>
      Promise.all([
        queryClient.invalidateQueries({ queryKey: ["get", "/customer/list"] }),
        queryClient.invalidateQueries({ queryKey: ["get", "/customer/info"] }),
      ]),
  });
};

export const useCreateCustomer = () => useCustomerMutation(createCustomer);
export const useUpdateCustomer = () => useCustomerMutation(updateCustomer);
export const useDeleteCustomers = () => useCustomerMutation(deleteCustomers);
export const useSetCustomerBlockedState = () => useCustomerMutation(setCustomerBlockedState);
