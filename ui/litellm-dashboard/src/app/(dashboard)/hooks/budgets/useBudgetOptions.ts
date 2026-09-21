"use client";

import { useQuery, type UseQueryResult } from "@tanstack/react-query";

import { apiClient } from "@/components/networking";

import { budgetKeys, type budgetItem } from "./useBudgets";

const BUDGET_OPTIONS_PATH = "/budget/list";

export const useBudgetOptions = (accessToken: string | null, enabled = true): UseQueryResult<budgetItem[]> => {
  const queryOptions = {
    queryKey: [...budgetKeys.all, "options"],
    queryFn: () => apiClient.get<budgetItem[]>(BUDGET_OPTIONS_PATH, { accessToken }),
    enabled: Boolean(accessToken) && enabled,
    staleTime: 60_000,
  };
  return useQuery(queryOptions);
};
