"use client";

import { useTranslation } from "react-i18next";
import AppDetailPage from "../_components/AppDetailPage";
import AppBudgetDefaults from "../_components/AppBudgetDefaults";
import CustomerBudgetCatalog, { budgetGroupCatalogs } from "../../customers/_components/CustomerBudgetCatalog";

const memberCatalog = budgetGroupCatalogs.find(({ scope }) => scope === "app-member")!;

export default function AppBudgetGroupsPage() {
  const { t } = useTranslation();
  return (
    <AppDetailPage active="budget-groups">
      {(app) => (
        <div key={app.app_id} className="w-full space-y-6">
          <p className="text-sm text-muted-foreground">{t("apps.sharedMemberGroupsHint")}</p>
          <CustomerBudgetCatalog catalog={memberCatalog} includeAppDefaults={false} />
          <div className="border-t pt-6">
            <AppBudgetDefaults scope="member" app={app} />
          </div>
        </div>
      )}
    </AppDetailPage>
  );
}
