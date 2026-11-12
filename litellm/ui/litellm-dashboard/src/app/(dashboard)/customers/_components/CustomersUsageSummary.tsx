"use client";

import { useMemo } from "react";
import { skipToken, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { defaultUsageDateRange } from "@/app/(dashboard)/usage/_components/usageUrlState";
import { dailyTotals, overviewTotals } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { Sparkline, Stat } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { dailyActivityAggregatedCall } from "@/components/networking";
import { ErrorState } from "@/components/shared/ErrorState";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  EMPTY_DAILY_ACTIVITY_RESPONSE,
  toDailyData,
} from "@/components/UsagePage/dailyActivityApi";

const BRAND = "#2b3fd6";

export default function CustomersUsageSummary() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const range = useMemo(() => defaultUsageDateRange("last_30_days").value, []);
  const startTime = range.from;
  const endTime = range.to;
  const request =
    accessToken && startTime && endTime ? { accessToken, startTime, endTime, includeEntityBreakdown: false } : null;
  const usage = useQuery({
    queryKey: ["customer-portal", "all-customers-daily-usage", accessToken, startTime?.getTime(), endTime?.getTime()],
    queryFn: request ? () => dailyActivityAggregatedCall("customer", request) : skipToken,
  });
  const data = usage.data ?? EMPTY_DAILY_ACTIVITY_RESPONSE;
  const trend = useMemo(() => dailyTotals(toDailyData(data)), [data]);
  const totals = useMemo(() => overviewTotals(data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA, null), [data]);

  return (
    <section
      aria-label={t("customerPortal.listSummary.title")}
      aria-busy={usage.isPending}
      className="overflow-hidden rounded-lg border bg-card"
    >
      <header className="flex items-center justify-between gap-3 border-b px-5 py-3">
        <h2 className="text-sm font-semibold">{t("customerPortal.listSummary.title")}</h2>
        <span className="text-xs text-muted-foreground">{t("shared.dateRangePresets.last30Days")}</span>
      </header>
      {usage.isError ? (
        <ErrorState title={t("customerPortal.listSummary.error")} onRetry={() => void usage.refetch()} />
      ) : (
        <div className="grid divide-y sm:grid-cols-2 sm:divide-x sm:divide-y-0">
          <div className="flex min-w-0 items-center justify-between gap-5 px-5 py-4">
            <Stat
              label={t("usage.statStrip.spend")}
              value={t("format.usd", { val: totals.spend })}
              pending={usage.isPending}
            />
            <Sparkline data={trend} dataKey="spend" color={BRAND} className="h-10 w-28 min-w-0 max-w-[35%]" />
          </div>
          <div className="flex min-w-0 items-center justify-between gap-5 px-5 py-4">
            <Stat
              label={t("usage.statStrip.totalTokens")}
              value={t("format.compactNumber", { val: totals.tokens })}
              exact={t("usage.statStrip.exactTokens", { tokens: t("format.number", { val: totals.tokens }) })}
              pending={usage.isPending}
            />
            <Sparkline data={trend} dataKey="tokens" color={BRAND} className="h-10 w-28 min-w-0 max-w-[35%]" />
          </div>
        </div>
      )}
    </section>
  );
}
