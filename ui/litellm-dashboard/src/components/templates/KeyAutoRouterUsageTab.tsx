"use client";

import React from "react";

import { AutoRouterUsageView } from "@/app/(dashboard)/cost-optimization/_components/AutoRouterBenchmarksTab";
import type { ActivityDateRange } from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";

interface KeyAutoRouterUsageTabProps {
  accessToken: string | null;
  keyToken: string;
  activity: ActivityDateRange;
}

const KeyAutoRouterUsageTab: React.FC<KeyAutoRouterUsageTabProps> = ({ accessToken, keyToken, activity }) => (
  <AutoRouterUsageView accessToken={accessToken} activity={activity} apiKey={keyToken} />
);

export default KeyAutoRouterUsageTab;
