"use client";

import BudgetPanel from "./components/budget_panel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Budgets() {
  const { accessToken } = useAuthorized();
  return <BudgetPanel accessToken={accessToken} />;
}
