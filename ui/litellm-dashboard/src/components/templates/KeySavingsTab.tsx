"use client";

import ScopedSavingsTab from "@/components/shared/ScopedSavingsTab";
import { hasProxyWideSpendView, spendScopeUserId } from "@/utils/roles";
import type { ActivityDateRange } from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";

interface KeySavingsTabProps {
  accessToken: string | null;
  keyToken: string;
  userId: string | null;
  userRole: string;
  activity: ActivityDateRange;
}

const KeySavingsTab = ({ accessToken, keyToken, userId, userRole, activity }: KeySavingsTabProps) => (
  <ScopedSavingsTab
    accessToken={accessToken}
    scope={{ userId: spendScopeUserId(userRole, userId), apiKey: keyToken }}
    activity={activity}
    entityType="key"
    scopeNote={
      hasProxyWideSpendView(userRole)
        ? undefined
        : "Showing your own requests on this key. A key shared across a team will have spend from other members that is not counted here."
    }
  />
);

export default KeySavingsTab;
