"use client";

import React from "react";

import { AutoRouterUsageView } from "@/app/(dashboard)/cost-optimization/_components/AutoRouterBenchmarksTab";
import { useActivityDateRange } from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";

interface KeyAutoRouterUsageTabProps {
  accessToken: string | null;
  keyToken: string;
}

const KeyAutoRouterUsageTab: React.FC<KeyAutoRouterUsageTabProps> = ({ accessToken, keyToken }) => (
  <AutoRouterUsageView accessToken={accessToken} activity={useActivityDateRange()} apiKey={keyToken} />
);

export default KeyAutoRouterUsageTab;
