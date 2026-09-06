"use client";

import CostOptimizationView from "./_components/CostOptimizationView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CostOptimizationPage() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <CostOptimizationView accessToken={accessToken} userId={userId} userRole={userRole} />;
}
