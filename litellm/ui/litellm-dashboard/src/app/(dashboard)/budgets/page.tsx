"use client";

import BudgetPanel from "./_components/BudgetPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Budgets() {
  const { accessToken } = useAuthorized();
  return <BudgetPanel accessToken={accessToken} />;
}
