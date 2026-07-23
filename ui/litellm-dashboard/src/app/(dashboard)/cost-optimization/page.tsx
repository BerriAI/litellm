"use client";

import UsageTab from "./_components/UsageTab";
import { useDailyActivityRange } from "./_components/useDailyActivityRange";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CostOptimizationPage() {
  const { accessToken, userId, userRole } = useAuthorized();
  const activity = useDailyActivityRange(accessToken, userId, userRole);
  return <UsageTab accessToken={accessToken} activity={activity} />;
}
