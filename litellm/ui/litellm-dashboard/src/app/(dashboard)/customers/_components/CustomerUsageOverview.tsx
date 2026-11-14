"use client";

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { skipToken, useQuery } from "@tanstack/react-query";
import { useQueryStates } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  dateRangeFromParams,
  dateRangePatch,
  defaultUsageDateRange,
  usageUrlParsers,
} from "@/app/(dashboard)/usage/_components/usageUrlState";
import { overviewTotals } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import UsageBreakdownChart from "@/app/(dashboard)/usage/_components/components/overview/UsageBreakdownChart";
import UsageStatStrip from "@/app/(dashboard)/usage/_components/components/overview/UsageStatStrip";
import { dailyActivityAggregatedCall } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { ErrorState } from "@/components/shared/ErrorState";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  EMPTY_DAILY_ACTIVITY_RESPONSE,
  toDailyData,
} from "@/components/UsagePage/dailyActivityApi";

const dateParsers = {
  range: usageUrlParsers.range,
  from: usageUrlParsers.from,
  to: usageUrlParsers.to,
};

export default function CustomerUsageOverview({ customerId }: { customerId: string }) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [params, setParams] = useQueryStates(dateParsers);
  const defaultRange = useMemo(() => defaultUsageDateRange("month_to_date"), []);
  const dateValue = useMemo(() => dateRangeFromParams(params) ?? defaultRange.value, [params, defaultRange]);
  const preset = params.range ?? (params.from === null && params.to === null ? defaultRange.preset : null);
  const startTime = dateValue.from;
  const endTime = dateValue.to;
  const request =
    accessToken && startTime && endTime
      ? { accessToken, startTime, endTime, entityIds: [customerId], includeEntityBreakdown: false }
      : null;
  const usage = useQuery({
    queryKey: ["customer-portal", "daily-usage", customerId, startTime?.getTime(), endTime?.getTime()],
    queryFn: request ? () => dailyActivityAggregatedCall("customer", request) : skipToken,
  });
  const data = usage.data ?? EMPTY_DAILY_ACTIVITY_RESPONSE;
  const results = useMemo(() => toDailyData(data), [data]);
  const totals = useMemo(() => overviewTotals(data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA, null), [data]);

  return (
    <section className="space-y-3" aria-label={t("customerPortal.usage.title")}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">{t("customerPortal.usage.title")}</h2>
          <p className="text-xs text-muted-foreground">{t("customerPortal.usage.description")}</p>
        </div>
        <AdvancedDatePicker
          value={dateValue}
          presetShortLabel={preset}
          showTimeRange={false}
          onValueChange={(value, selectedPreset) => void setParams(dateRangePatch(value, selectedPreset))}
        />
      </div>
      {usage.isError ? (
        <ErrorState title={t("customerPortal.usage.error")} onRetry={() => void usage.refetch()} />
      ) : (
        <>
          <UsageStatStrip
            results={results}
            totals={totals}
            loading={usage.isPending}
            requestCountsPending={usage.isPending}
            budget={null}
            showBudgetHint={false}
          />
          <UsageBreakdownChart results={results} loading={usage.isPending} />
        </>
      )}
    </section>
  );
}
