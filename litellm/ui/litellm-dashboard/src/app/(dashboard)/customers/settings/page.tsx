"use client";

import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { $api } from "@/lib/http/api";
import { ErrorState } from "@/components/shared/ErrorState";
import type { BudgetGroups } from "../_components/customerBudgetGroupsPlan";
import CustomerBudgetGroups from "../_components/CustomerBudgetGroups";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { uiHref } from "@/utils/uiHref";

const budgetGroupTab = parseAsStringLiteral(["organization", "app"])
  .withDefault("organization")
  .withOptions({ history: "push" });

const catalogs = [
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
] as const;

function Catalog({ catalog }: { catalog: (typeof catalogs)[number] }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { scope, endpoint, invalidate } = catalog;
  const groups = $api.useQuery("get", endpoint, {});
  const saved = (data: BudgetGroups) => {
    queryClient.setQueryData($api.queryOptions("get", endpoint, {}).queryKey, data);
    for (const key of invalidate) void queryClient.invalidateQueries({ queryKey: [key] });
  };
  return (
    <>
      {groups.data && <CustomerBudgetGroups global scope={scope} organization={groups.data} onSaved={saved} />}
      {groups.isError && <ErrorState onRetry={() => void groups.refetch()} />}
      {groups.isPending && <p>{t("customerPortal.common.loading")}</p>}
    </>
  );
}

export default function CustomerSettings() {
  const { t } = useTranslation();
  const [tab, setTab] = useQueryState("tab", budgetGroupTab);
  const changeTab = (value: unknown) => {
    if (value === "organization" || value === "app") void setTab(value);
  };
  const { accessToken, userRole } = useAuthorized();
  const canManage = Boolean(accessToken && isProxyAdminRole(userRole));
  if (!canManage) return <ErrorState message={t("customerPortal.hierarchy.adminOnly")} />;
  return (
    <main className="space-y-5 px-6 py-5">
      <h1 className="text-lg font-semibold">{t("customerPortal.budgetGroups.title")}</h1>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">{t("customerPortal.budgetGroups.settingsHint")}</p>
        <Button variant="outline" nativeButton={false} render={<a href={uiHref("apps")} />}>
          {t("apps.definition.title")}
        </Button>
      </div>
      <Tabs value={tab} onValueChange={changeTab} className="gap-6">
        <div className="overflow-x-auto border-b">
          <TabsList variant="line" aria-label={t("customerPortal.budgetGroups.title")} className="h-auto gap-6 p-0">
            {catalogs.map(({ scope, label }) => (
              <TabsTrigger
                key={scope}
                value={scope}
                className="flex-none rounded-none px-0 py-3 group-data-horizontal/tabs:after:bottom-0"
              >
                {t(label)}
              </TabsTrigger>
            ))}
          </TabsList>
        </div>
        {catalogs.map((catalog) => (
          <TabsContent key={catalog.scope} value={catalog.scope} keepMounted>
            <Catalog catalog={catalog} />
          </TabsContent>
        ))}
      </Tabs>
    </main>
  );
}
