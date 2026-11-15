"use client";

import { useTranslation } from "react-i18next";
import ScopedSavingsTab from "@/components/shared/ScopedSavingsTab";
import { reportingKeyId } from "@/components/UsagePage/keyActivityData";
import { hasProxyWideSpendView, spendScopeUserId } from "@/utils/roles";
import type { ActivityDateRange } from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";

interface KeySavingsTabProps {
  accessToken: string | null;
  keyToken: string;
  userId: string | null;
  userRole: string;
  activity: ActivityDateRange;
}

const KeySavingsTab = ({ accessToken, keyToken, userId, userRole, activity }: KeySavingsTabProps) => {
  const { t } = useTranslation();
  return (
    <ScopedSavingsTab
      accessToken={accessToken}
      scope={{ userId: spendScopeUserId(userRole, userId), apiKey: reportingKeyId(keyToken) }}
      activity={activity}
      entityType="key"
      scopeNote={hasProxyWideSpendView(userRole) ? undefined : t("keyDetail.keySavingsTab.scopeNote")}
    />
  );
};

export default KeySavingsTab;
