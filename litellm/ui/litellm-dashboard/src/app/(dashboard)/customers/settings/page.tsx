"use client";

import { Settings } from "lucide-react";

import { useTranslation } from "react-i18next";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { ErrorState } from "@/components/shared/ErrorState";
import { PageHeader, PageHeaderTitle, PageHeaderDescription } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { uiHref } from "@/utils/uiHref";

import CustomerBudgetCatalog, { budgetGroupCatalogs } from "../_components/CustomerBudgetCatalog";

const catalogs = budgetGroupCatalogs.filter(({ scope }) => scope !== "app-member");
const budgetGroupTab = parseAsStringLiteral(catalogs.map(({ scope }) => scope))
  .withDefault("organization")
  .withOptions({ history: "push" });

export default function CustomerSettings() {
  const { t } = useTranslation();
  const [tab, setTab] = useQueryState("tab", budgetGroupTab);
  const changeTab = (value: unknown) => {
    const selected = catalogs.find(({ scope }) => scope === value);
    if (selected) void setTab(selected.scope);
  };
  const { accessToken, userRole } = useAuthorized();
  const canManage = Boolean(accessToken && isProxyAdminRole(userRole));
  if (!canManage) return <ErrorState message={t("customerPortal.hierarchy.adminOnly")} />;
  return (
    <>
      <PageHeader>
        <PageHeaderTitle>
          <Settings />
          {t("customerPortal.budgetGroups.title")}
        </PageHeaderTitle>
        <div className="mt-1.5 flex flex-wrap items-center justify-between gap-3">
          <PageHeaderDescription className="mt-0">
            {t("customerPortal.budgetGroups.settingsHint")}
          </PageHeaderDescription>
          <Button variant="outline" nativeButton={false} render={<a href={uiHref("apps")} />}>
            {t("apps.definition.title")}
          </Button>
        </div>
      </PageHeader>
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
            <CustomerBudgetCatalog catalog={catalog} />
          </TabsContent>
        ))}
      </Tabs>
    </>
  );
}
