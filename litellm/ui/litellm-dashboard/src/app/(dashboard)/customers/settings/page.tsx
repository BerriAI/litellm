"use client";

import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { $api } from "@/lib/http/api";
import { ErrorState } from "@/components/shared/ErrorState";
import type { BudgetGroups } from "../_components/customerBudgetGroupsPlan";
import CustomerBudgetGroups from "../_components/CustomerBudgetGroups";

export default function CustomerSettings() {
  const { t } = useTranslation();
  const { accessToken, userRole } = useAuthorized();
  const queryClient = useQueryClient();
  const canManage = Boolean(accessToken && isProxyAdminRole(userRole));
  const groups = $api.useQuery("get", "/management/v1/customer_organization_budget_groups", {}, { enabled: canManage });
  const saved = (data: BudgetGroups) => {
    queryClient.setQueryData(
      $api.queryOptions("get", "/management/v1/customer_organization_budget_groups", {}).queryKey,
      data,
    );
    void queryClient.invalidateQueries({ queryKey: ["customer-hierarchy"] });
    void queryClient.invalidateQueries({ queryKey: ["customer-organizations"] });
  };
  if (!canManage) return <ErrorState message={t("customerPortal.hierarchy.adminOnly")} />;
  return (
    <main className="space-y-5 px-6 py-5">
      <h1 className="text-lg font-semibold">{t("customerPortal.organizationGroups.title")}</h1>
      {groups.data && <CustomerBudgetGroups global organization={groups.data} onSaved={saved} />}
      {groups.isError && <ErrorState onRetry={() => void groups.refetch()} />}
      {groups.isPending && <p>{t("customerPortal.common.loading")}</p>}
    </main>
  );
}
