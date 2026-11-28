"use client";

import { type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { DailyData } from "@/components/UsagePage/types";
import { KeyRound, Users } from "lucide-react";
import type { OverviewTotals } from "./overviewData";
import { Panel } from "./Primitives";
import UsageStatStrip from "./UsageStatStrip";
import UsageBreakdownChart from "./UsageBreakdownChart";
import SpendDisplayCurrency from "./SpendDisplayCurrency";

interface UsageOverviewProps {
  results: readonly DailyData[];
  totals: OverviewTotals;
  loading: boolean;
  requestCountsPending: boolean;
  budget: number | null;
  budgetPending?: boolean;
  subscriptionCosts: ReactNode;
  topKeys: ReactNode;
  topUsers: ReactNode;
  gatewayByEndpoint: ReactNode;
  topAgents: ReactNode;
  providerBreakdown: ReactNode;
}

export default function UsageOverview({
  results,
  totals,
  loading,
  requestCountsPending,
  subscriptionCosts,
  topKeys,
  topUsers,
  gatewayByEndpoint,
  topAgents,
  providerBreakdown,
  budget,
  budgetPending = false,
}: UsageOverviewProps) {
  const { t } = useTranslation();
  return (
    <div className="grid gap-3">
      <UsageStatStrip
        results={results}
        totals={totals}
        loading={loading}
        requestCountsPending={requestCountsPending}
        budget={budget}
        showBudgetHint={!budgetPending}
        spendDetails={<SpendDisplayCurrency usdSpend={totals.spend} loading={loading} />}
      />

      {subscriptionCosts}

      <UsageBreakdownChart results={results} loading={loading} />

      {topAgents}
      {providerBreakdown}

      <Panel icon={KeyRound} title={t("usage.overview.topVirtualKeys")}>
        {topKeys}
      </Panel>
      <Panel icon={Users} title={t("usage.overview.topUsersBySpend")}>
        {topUsers}
      </Panel>
      {gatewayByEndpoint}
    </div>
  );
}
