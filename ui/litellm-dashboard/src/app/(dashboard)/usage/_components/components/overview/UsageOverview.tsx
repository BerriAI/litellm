"use client";

/**
 * Usage overview: a stat strip, the Model Leaderboard's stacked chart and
 * tooltip applied to the usage aggregate, then a ranked share list in
 * matching colors.
 */

import { type ReactNode } from "react";
import type { DailyData } from "@/components/UsagePage/types";
import { KeyRound, Users } from "lucide-react";
import type { OverviewTotals } from "./overviewData";
import { Panel } from "./Primitives";
import UsageStatStrip from "./UsageStatStrip";
import UsageBreakdownChart from "./UsageBreakdownChart";

interface UsageOverviewProps {
  results: readonly DailyData[];
  totals: OverviewTotals;
  loading: boolean;
  requestCountsPending: boolean;
  budget: number | null;
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
  topKeys,
  topUsers,
  gatewayByEndpoint,
  topAgents,
  providerBreakdown,
  budget,
}: UsageOverviewProps) {
  return (
    <div className="grid gap-3">
      <UsageStatStrip
        results={results}
        totals={totals}
        loading={loading}
        requestCountsPending={requestCountsPending}
        budget={budget}
      />

      <UsageBreakdownChart results={results} loading={loading} />

      {topAgents}
      {providerBreakdown}

      <Panel icon={KeyRound} title="Top Virtual Keys">
        {topKeys}
      </Panel>
      <Panel icon={Users} title="Top Users by Spend">
        {topUsers}
      </Panel>
      {gatewayByEndpoint}
    </div>
  );
}
