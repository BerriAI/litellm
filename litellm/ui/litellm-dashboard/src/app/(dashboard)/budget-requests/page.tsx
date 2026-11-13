"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import BudgetRequests from "./_components/BudgetRequests";

export default function BudgetRequestsPage() {
  const { accessToken, userRole, userId, isViewOnly } = useAuthorized();
  return accessToken ? (
    <BudgetRequests
      key={`${accessToken}:${userRole}:${userId}`}
      accessToken={accessToken}
      userRole={userRole ?? ""}
      userId={userId ?? ""}
      isViewOnly={isViewOnly}
    />
  ) : null;
}
