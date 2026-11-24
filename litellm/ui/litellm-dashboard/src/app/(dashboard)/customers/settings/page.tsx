"use client";

import { Settings } from "lucide-react";

import { useTranslation } from "react-i18next";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { ErrorState } from "@/components/shared/ErrorState";
import { PageContent, PageTabs, PageTabsContent, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { PageHeader, PageHeaderTitle, PageHeaderDescription, PageHeaderControls } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
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
      <PageTabs value={tab} onValueChange={changeTab}>
        <PageHeader>
          <PageHeaderTitle>
            <Settings />
            {t("customerPortal.budgetGroups.title")}
          </PageHeaderTitle>
          <PageHeaderDescription>{t("customerPortal.budgetGroups.settingsHint")}</PageHeaderDescription>
          <PageHeaderControls className="flex-wrap">
            <PageTabsList aria-label={t("customerPortal.budgetGroups.title")} className="min-w-0 basis-full sm:flex-1">
              {catalogs.map(({ scope, label }) => (
                <PageTabsTrigger key={scope} value={scope}>
                  {t(label)}
                </PageTabsTrigger>
              ))}
            </PageTabsList>
            <Button variant="outline" nativeButton={false} render={<a href={uiHref("apps")} />}>
              {t("apps.definition.title")}
            </Button>
          </PageHeaderControls>
        </PageHeader>
        <PageContent>
          {catalogs.map((catalog) => (
            <PageTabsContent key={catalog.scope} value={catalog.scope} keepMounted>
              <CustomerBudgetCatalog catalog={catalog} />
            </PageTabsContent>
          ))}
        </PageContent>
      </PageTabs>
    </>
  );
}
