import { useQuery, useMutation, useQueryClient, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { getBudgetList, budgetCreateCall, budgetUpdateCall, budgetDeleteCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { budgetItem } from "@/components/budgets/budget_panel";

export const budgetKeys = createQueryKeys("budgets");

export const useBudgets = (): UseQueryResult<budgetItem[]> => {
  const { accessToken } = useAuthorized();
  return useQuery<budgetItem[]>({
    queryKey: budgetKeys.list({}),
    queryFn: async () => {
      const data = await getBudgetList(accessToken!);
      return (data ?? []).filter((item: budgetItem | null): item is budgetItem => item != null);
    },
    enabled: Boolean(accessToken),
  });
};

export const useCreateBudget = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<unknown, Error, Record<string, any>>({
    mutationFn: async (formValues) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return budgetCreateCall(accessToken, formValues);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: budgetKeys.all });
    },
  });
};

export const useUpdateBudget = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<unknown, Error, Record<string, any>>({
    mutationFn: async (formValues) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return budgetUpdateCall(accessToken, formValues);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: budgetKeys.all });
    },
  });
};

export const useDeleteBudget = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<unknown, Error, string>({
    mutationFn: async (budgetId) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return budgetDeleteCall(accessToken, budgetId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: budgetKeys.all });
    },
  });
};
