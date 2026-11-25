"use client";

import AppDetailPage from "../_components/AppDetailPage";
import CustomerBudgetCatalog, { budgetGroupCatalogs } from "../../customers/_components/CustomerBudgetCatalog";

const memberCatalog = budgetGroupCatalogs.find(({ scope }) => scope === "app-member")!;

export default function AppBudgetGroupsPage() {
  return (
    <AppDetailPage active="budget-groups">
      {(app) => <CustomerBudgetCatalog key={app.app_id} catalog={memberCatalog} app={app} />}
    </AppDetailPage>
  );
}
