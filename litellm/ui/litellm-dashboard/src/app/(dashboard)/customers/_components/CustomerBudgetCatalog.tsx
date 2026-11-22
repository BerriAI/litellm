"use client";

import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import { $api } from "@/lib/http/api";
import { ErrorState } from "@/components/shared/ErrorState";
import type { BudgetGroups } from "./customerBudgetGroupsPlan";
import CustomerBudgetGroups from "./CustomerBudgetGroups";
import AppBudgetDefaults from "../../apps/_components/AppBudgetDefaults";
import type { App } from "../../apps/_components/appForms";
import { invalidateAppBudgets } from "../../apps/_components/appApi";

export const budgetGroupCatalogs = [
  {
    scope: "overall-app",
    label: "customerPortal.budgetGroups.overallAppTab",
    endpoint: "/management/v1/app_budget_groups",
    invalidate: ["apps", "app"],
  },
  {
    scope: "app-group",
    label: "customerPortal.budgetGroups.appGroupTab",
    endpoint: "/management/v1/app_group_budget_groups",
    invalidate: ["app-groups", "app-group-organizations"],
  },
  {
    scope: "organization",
    label: "customerPortal.budgetGroups.overallTab",
    endpoint: "/management/v1/customer_organization_budget_groups",
    invalidate: ["customer-hierarchy", "customer-organizations"],
  },
  {
    scope: "app",
    label: "customerPortal.budgetGroups.appTab",
    endpoint: "/management/v1/app_customer_organization_budget_groups",
    invalidate: ["apps", "app-organization", "app-member"],
  },
  {
    scope: "app-member",
    label: "customerPortal.budgetGroups.appMemberTab",
    endpoint: "/management/v1/app_customer_member_budget_groups",
    invalidate: ["apps", "app", "app-organization", "app-member"],
  },
] as const;

export type BudgetGroupCatalogScope = (typeof budgetGroupCatalogs)[number]["scope"];

export default function CustomerBudgetCatalog({
  catalog,
  app,
}: {
  catalog: (typeof budgetGroupCatalogs)[number];
  app?: App;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { scope, endpoint, invalidate } = catalog;
  const groups = $api.useQuery("get", endpoint, {});
  const saved = (data: BudgetGroups) => {
    queryClient.setQueryData($api.queryOptions("get", endpoint, {}).queryKey, data);
    for (const key of invalidate) void queryClient.invalidateQueries({ queryKey: [key] });
    if (scope !== "organization") void invalidateAppBudgets(queryClient);
  };
  return (
    <>
      {groups.data && <CustomerBudgetGroups global scope={scope} organization={groups.data} onSaved={saved} />}
      {groups.isError && <ErrorState onRetry={() => void groups.refetch()} />}
      {groups.isPending && <p>{t("customerPortal.common.loading")}</p>}
      {scope === "app" && <AppBudgetDefaults app={app} />}
      {scope === "app-member" && <AppBudgetDefaults scope="member" app={app} />}
    </>
  );
}
