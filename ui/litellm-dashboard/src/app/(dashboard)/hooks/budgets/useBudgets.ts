"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { SortingState } from "@tanstack/react-table";
import { useCallback } from "react";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { apiClient, budgetCreateCall, budgetUpdateCall, budgetDeleteCall } from "@/components/networking";

import { createQueryKeys } from "../common/queryKeysFactory";
import {
  useResourceList,
  type ResourceListPage,
  type ResourceListQuery,
  type ResourceListResult,
} from "../common/useResourceList";
import { serializeBudgetFilters } from "./budgetFilters";

export interface budgetItem {
  budget_id: string;
  max_budget: number | null;
  soft_budget: number | null;
  rpm_limit: number | null;
  tpm_limit: number | null;
  budget_duration: string | null;
  budget_reset_at: string | null;
  created_at: string;
  updated_at: string;
}

export const BUDGET_LIST_PATH = "/management/v1/budgets";

export const budgetKeys = createQueryKeys("budgets");

const DEFAULT_PAGE_SIZE = 50;
const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

export const useBudgetList = (): ResourceListResult<budgetItem> => {
  const { accessToken } = useAuthorized();

  const fetchPage = useCallback(
    (query: ResourceListQuery, signal: AbortSignal): Promise<ResourceListPage<budgetItem>> =>
      apiClient.get<ResourceListPage<budgetItem>>(BUDGET_LIST_PATH, { accessToken, query, signal }),
    [accessToken],
  );

  const listOptions = {
    queryKey: budgetKeys.lists(),
    fetchPage,
    serializeFilters: serializeBudgetFilters,
    defaultSorting: DEFAULT_SORTING,
    defaultPageSize: DEFAULT_PAGE_SIZE,
    enabled: Boolean(accessToken),
  };
  return useResourceList<budgetItem>(listOptions);
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
