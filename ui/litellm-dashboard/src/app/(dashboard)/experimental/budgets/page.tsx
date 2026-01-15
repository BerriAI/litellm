"use client";

import BudgetPanel from "@/components/budgets/budget_panel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const BudgetsPage = () => {
  const { accessToken } = useAuthorized();

  return <BudgetPanel accessToken={accessToken} />;
};

export default BudgetsPage;
