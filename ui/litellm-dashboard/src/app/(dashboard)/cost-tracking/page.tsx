"use client";

import { CostTrackingSettings } from "./components";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CostTracking() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <CostTrackingSettings userID={userId} userRole={userRole} accessToken={accessToken} />;
}
