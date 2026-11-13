"use client";

import { useMutation, useQuery, useQueryClient, type QueryFunctionContext } from "@tanstack/react-query";
import useAuthorized from "../useAuthorized";
import { apiClient } from "@/components/networking";
import { budgetKeys } from "./useBudgets";
import type { BudgetAssignment, BudgetAssignmentPatch, BudgetSubject } from "./budgetProfileTypes";

export function useBudgetAssignment(subject: BudgetSubject) {
  const { accessToken, userRole, isLoading, isAuthorized } = useAuthorized();
  const queryClient = useQueryClient();
  const path = `/management/v1/budget_assignments/${subject.type}/${encodeURIComponent(subject.id)}`;
  const queryKey = ["budget-assignment", accessToken, userRole, subject.type, subject.id] as const;
  const queryOptions = {
    queryKey,
    queryFn: ({ signal }: QueryFunctionContext<typeof queryKey>) =>
      apiClient.get<BudgetAssignment>(path, { accessToken, signal }),
    enabled: Boolean(accessToken) && !isLoading && isAuthorized,
    refetchInterval: 30_000,
  };
  const query = useQuery(queryOptions);
  const update = useMutation({
    mutationFn: (body: BudgetAssignmentPatch) => apiClient.patch<BudgetAssignment>(path, { accessToken, body }),
    onSuccess: async (data) => {
      await queryClient.cancelQueries({ queryKey, exact: true });
      queryClient.setQueryData(queryKey, data);
      await queryClient.invalidateQueries({ queryKey: budgetKeys.all });
    },
  });
  return { query, update };
}
