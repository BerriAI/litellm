"use client";

import React, { useMemo, useState } from "react";
import { Collapse } from "antd";

import { AreaChart, DonutChart } from "@/components/shared/charts";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { userDailyActivityCall } from "@/components/networking";
import { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { all_admin_roles } from "@/utils/roles";
import { usePaginatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/usePaginatedDailyActivity";

interface UsageTabProps {
  accessToken: string | null;
  userId: string | null;
  userRole: string;
}

type DateRange = { from?: Date; to?: Date };

const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

const usd = (value: number): string => {
  const decimals = value > 0 && value < 1 ? 4 : 2;
  return `$${formatNumberWithCommas(value, decimals)}`;
};

const shortDate = (iso: string): string =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });

const compressionOf = (m: SpendMetrics): number => m.compression_savings_spend ?? 0;
const cachingOf = (m: SpendMetrics): number => m.prompt_caching_savings_spend ?? 0;
const savedTokensOf = (m: SpendMetrics): number => m.compression_saved_tokens ?? 0;
const autorouterOf = (m: SpendMetrics): number => m.autorouter_savings_spend ?? 0;
const autorouterRequestsOf = (m: SpendMetrics): number => m.autorouter_requests ?? 0;
const autorouterEscalatedOf = (m: SpendMetrics): number => m.autorouter_escalated_requests ?? 0;

const pct = (value: number): string => `${formatNumberWithCommas(value * 100, 1)}%`;

const MethodologyNote = () => (
  <Collapse
    ghost
    items={[
      {
        key: "methodology",
        label: <span className="text-sm font-medium">How savings are calculated</span>,
        children: (
          <div className="space-y-3 text-sm text-muted-foreground">
            <p>
              Savings are computed for each request when it is logged, using the provider&apos;s reported usage and the
              model&apos;s pricing, then summed into a daily rollup. Totals below are read from that rollup over the
              selected date range, so the numbers never require a scan of raw request logs.
            </p>
            <p>
              Compression savings are the tokens Headroom removed before the call, priced at the model&apos;s input
              rate: <code>compression_saved_tokens * input_cost_per_token</code>
            </p>
            <p>
              Prompt caching savings are the tokens the provider served from cache (Anthropic{" "}
              <code>cache_read_input_tokens</code>, or OpenAI-style <code>prompt_tokens_details.cached_tokens</code>),
              priced at the discount between the normal input rate and the cache-read rate:{" "}
              <code>cache_read_input_tokens * max(input_cost_per_token - cache_read_input_token_cost, 0)</code>
            </p>
            <p>
              Autorouter savings are an estimate: for each request the complexity router handled, the same token usage
              is repriced against the most expensive model it could have routed to, and the difference from the actual
              spend is credited as saved (clamped at zero). The baseline model is not called, so this is a
              counterfactual rather than a realized discount. Escalation rate is the share of routed requests that asked
              to escalate to a stronger model, a quality signal alongside the savings.
            </p>
            <p>
              Total saved is the sum of the drivers. Models without a separate cache-read price in the pricing map
              contribute zero caching savings rather than erroring.
            </p>
          </div>
        ),
      },
    ]}
  />
);

const SummaryCard = ({ label, value, hint }: { label: string; value: string; hint?: string }) => (
  <Card>
    <CardHeader>
      <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
    </CardHeader>
    <CardContent>
      <p className="text-2xl font-semibold text-foreground">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </CardContent>
  </Card>
);

const UsageTab: React.FC<UsageTabProps> = ({ accessToken, userId, userRole }) => {
  const initialFrom = useMemo(() => new Date(new Date().getTime() - THIRTY_DAYS_MS), []);
  const initialTo = useMemo(() => new Date(), []);
  const [dateValue, setDateValue] = useState<DateRange>({ from: initialFrom, to: initialTo });

  const startTime = dateValue.from ?? null;
  const endTime = dateValue.to ?? null;
  const isAdmin = all_admin_roles.includes(userRole);
  const effectiveUserId = isAdmin ? null : userId;

  const { data, loading, isFetchingMore } = usePaginatedDailyActivity({
    fetchFn: userDailyActivityCall,
    args: [accessToken, startTime, endTime, effectiveUserId],
    enabled: !!accessToken && !!startTime && !!endTime,
  });

  const results = data.results as DailyData[];

  const compressionTotal = useMemo(() => results.reduce((sum, d) => sum + compressionOf(d.metrics), 0), [results]);
  const cachingTotal = useMemo(() => results.reduce((sum, d) => sum + cachingOf(d.metrics), 0), [results]);
  const savedTokensTotal = useMemo(() => results.reduce((sum, d) => sum + savedTokensOf(d.metrics), 0), [results]);
  const autorouterTotal = useMemo(() => results.reduce((sum, d) => sum + autorouterOf(d.metrics), 0), [results]);
  const autorouterRequestsTotal = useMemo(
    () => results.reduce((sum, d) => sum + autorouterRequestsOf(d.metrics), 0),
    [results],
  );
  const autorouterEscalatedTotal = useMemo(
    () => results.reduce((sum, d) => sum + autorouterEscalatedOf(d.metrics), 0),
    [results],
  );
  const escalationRate = autorouterRequestsTotal > 0 ? autorouterEscalatedTotal / autorouterRequestsTotal : 0;
  const totalSaved = compressionTotal + cachingTotal + autorouterTotal;

  const overTime = useMemo(
    () =>
      results.map((d) => ({
        date: shortDate(d.date),
        Compression: compressionOf(d.metrics),
        "Prompt caching": cachingOf(d.metrics),
      })),
    [results],
  );

  const byDriver = useMemo(
    () =>
      [
        { driver: "Autorouter (est.)", usd: autorouterTotal },
        { driver: "Compression", usd: compressionTotal },
        { driver: "Prompt caching", usd: cachingTotal },
      ].filter((d) => d.usd > 0),
    [autorouterTotal, compressionTotal, cachingTotal],
  );

  return (
    <div className="w-full space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <MethodologyNote />
        <AdvancedDatePicker value={dateValue} onValueChange={(v) => setDateValue(v)} />
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <SummaryCard
          label="Total saved"
          value={usd(totalSaved)}
          hint={loading || isFetchingMore ? "Loading..." : "Autorouter (est.) + compression + prompt caching"}
        />
        <SummaryCard
          label="Compression savings"
          value={usd(compressionTotal)}
          hint={`${formatNumberWithCommas(savedTokensTotal)} tokens compressed`}
        />
        <SummaryCard label="Prompt caching savings" value={usd(cachingTotal)} hint="Cache read discount" />
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <SummaryCard
          label="Autorouter saved"
          value={usd(autorouterTotal)}
          hint="Estimated vs the most expensive configured model"
        />
        <SummaryCard
          label="Escalation rate"
          value={autorouterRequestsTotal > 0 ? pct(escalationRate) : "\u2014"}
          hint={
            autorouterRequestsTotal > 0
              ? `${formatNumberWithCommas(autorouterEscalatedTotal)} of ${formatNumberWithCommas(
                  autorouterRequestsTotal,
                )} routed requests asked to escalate`
              : "No autorouter requests yet"
          }
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Savings over time</CardTitle>
          </CardHeader>
          <CardContent>
            <AreaChart
              data={overTime}
              index="date"
              categories={["Compression", "Prompt caching"]}
              colors={["emerald", "blue"]}
              valueFormatter={usd}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Savings by driver</CardTitle>
          </CardHeader>
          <CardContent>
            <DonutChart
              className="h-80"
              data={byDriver}
              index="driver"
              category="usd"
              colors={["amber", "emerald", "blue"]}
              valueFormatter={usd}
              showLabel
              label={usd(totalSaved)}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default UsageTab;
