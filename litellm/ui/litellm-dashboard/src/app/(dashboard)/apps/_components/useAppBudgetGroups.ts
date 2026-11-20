"use client";

import { $api } from "@/lib/http/api";

export function useAppBudgetGroups(existingIds: (string | null | undefined)[] = [], enabled = true) {
  const groups = $api.useQuery("get", "/management/v1/app_customer_organization_budget_groups", {}, { enabled });
  const missingIds = existingIds.filter(
    (id): id is string => Boolean(id) && !groups.data?.default_budgets.some((group) => group.budget_id === id),
  );
  const needsLegacy = Boolean(groups.data) && missingIds.length > 0;
  const legacy = $api.useQuery(
    "get",
    "/management/v1/customer_organization_budget_groups",
    {},
    { enabled: enabled && needsLegacy },
  );
  return {
    isPending: groups.isPending || (needsLegacy && legacy.isPending),
    isError: groups.isError || (needsLegacy && legacy.isError),
    isSuccess: groups.isSuccess && (!needsLegacy || legacy.isSuccess),
    error: groups.error ?? (needsLegacy ? legacy.error : null),
    refetch: () => Promise.all([groups.refetch(), ...(needsLegacy ? [legacy.refetch()] : [])]),
    data:
      groups.data && (!needsLegacy || legacy.data)
        ? {
            ...groups.data,
            default_budgets: [
              ...groups.data.default_budgets,
              ...(legacy.data?.default_budgets ?? []).filter((group) => missingIds.includes(group.budget_id)),
            ],
          }
        : undefined,
  };
}
