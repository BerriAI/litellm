"use client";

import UsageTab from "./_components/UsageTab";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CostOptimizationPage() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <UsageTab accessToken={accessToken} userId={userId} userRole={userRole} />;
}
