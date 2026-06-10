"use client";

import { CostTrackingSettings } from "@/components/CostTrackingSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CostTrackingRoute() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <CostTrackingSettings userID={userId} userRole={userRole} accessToken={accessToken} />;
}
