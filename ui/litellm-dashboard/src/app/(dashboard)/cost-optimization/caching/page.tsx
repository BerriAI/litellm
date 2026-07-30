"use client";

import PromptCachingTab from "@/app/(dashboard)/cost-optimization/_components/PromptCachingTab";
import { useDailyActivityRange } from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function PromptCachingPage() {
  const { accessToken, userId, userRole } = useAuthorized();
  const activity = useDailyActivityRange(accessToken, userId, userRole);
  return <PromptCachingTab accessToken={accessToken} activity={activity} />;
}
