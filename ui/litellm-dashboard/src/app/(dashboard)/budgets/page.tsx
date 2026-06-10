"use client";

import BudgetPanel from "@/components/budgets/budget_panel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function BudgetsRoute() {
  const { accessToken } = useAuthorized();
  return <BudgetPanel accessToken={accessToken} />;
}
