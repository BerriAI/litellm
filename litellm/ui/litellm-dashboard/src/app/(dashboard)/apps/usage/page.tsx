"use client";

import { useMemo } from "react";
import AppDetailPage from "../_components/AppDetailPage";
import AppCustomerSpendTable from "../_components/AppCustomerSpendTable";
import type { App } from "../_components/appForms";
import { skipToken, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import useAuthorized from "../../hooks/useAuthorized";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { ErrorState } from "@/components/shared/ErrorState";
import UsageBreakdownChart from "../../usage/_components/components/overview/UsageBreakdownChart";
import { overviewTotals } from "../../usage/_components/components/overview/overviewData";
import { CustomerUsageDatePicker, CustomerUsageStats } from "../../customers/_components/CustomerUsageSummary";
import { customerAppUsageCall } from "../../customers/_components/customerUsageApi";
import { customerUsageTotals } from "../../customers/_components/customerUsageTotals";
import {
  customerEntityRankings,
  mergeCustomerActivityPages,
  rankCustomerUsage,
} from "../../customers/_components/customerUsageData";
import {
  calendarDate,
  customerPresetRange,
  useCustomerUsageRange,
} from "../../customers/_components/customerUsageRange";

const EMPTY_USAGE = mergeCustomerActivityPages([]);

export default function AppUsagePage() {
  return <AppDetailPage active="usage">{(app) => <AppUsage key={app.app_id} app={app} />}</AppDetailPage>;
}

function useAppUsage(appId: string, startDate: string, endDate: string) {
  const { accessToken, userId } = useAuthorized();
  return useQuery({
    queryKey: ["app-customer-usage", userId, appId, startDate, endDate],
    queryFn: accessToken
      ? ({ signal }) => {
          const request = { accessToken, appId, startDate, endDate, signal };
          return customerAppUsageCall(request);
        }
      : skipToken,
  });
}

function AppUsage({ app }: { app: App }) {
  const { t } = useTranslation();
  const { dateValue, onDateChange, preset } = useCustomerUsageRange();
  const startDate = calendarDate(dateValue.from);
  const endDate = calendarDate(dateValue.to);
  const usage = useAppUsage(app.app_id, startDate, endDate);
  const month = customerPresetRange("month");
  const monthly = useAppUsage(app.app_id, calendarDate(month.from), calendarDate(month.to));
  const monthlyTotals = useMemo(() => monthly.data && customerUsageTotals(monthly.data.results), [monthly.data]);
  const data = usage.data ?? EMPTY_USAGE;
  const totals = useMemo(() => overviewTotals(data.metadata, null), [data]);
  const selectedTotals = useMemo(() => customerUsageTotals(data.results), [data]);
  const rows = useMemo(() => {
    const labels = new Map(
      data.results.flatMap((day) =>
        Object.entries(day.breakdown.entities)
          .filter(([id]) => id !== "")
          .map(
            ([id, { metadata }]) =>
              [id, "alias" in metadata && typeof metadata.alias === "string" ? metadata.alias : id] as const,
          ),
      ),
    );
    const customers = customerEntityRankings(data.results, labels, data.metadata, {
      unassignedLabel: t("apps.usage.unassigned"),
    });
    return rankCustomerUsage(customers, "spend", totals);
  }, [data, totals, t]);
  const failed = usage.isError;
  const loading = usage.isPending;

  return (
    <>
      <header className="flex flex-wrap items-end justify-between gap-4">
        <p className="text-sm text-muted-foreground">{t("apps.usage.hint")}</p>
        <CustomerUsageDatePicker value={dateValue} onValueChange={onDateChange} presetShortLabel={preset} />
      </header>
      {failed && <ErrorState message={extractProxyErrorMessage(usage.error)} onRetry={() => void usage.refetch()} />}
      {!failed && (
        <>
          <section aria-label={t("apps.usage.totals")}>
            <CustomerUsageStats
              results={data.results}
              totals={totals}
              loading={loading}
              surcharge={data.metadata.total_customer_margin_spend}
            />
          </section>
          <section className="space-y-4">
            <h2 className="text-base font-medium">{t("apps.usage.customers")}</h2>
            {monthly.isError && !usage.isError && (
              <ErrorState message={extractProxyErrorMessage(monthly.error)} onRetry={() => void monthly.refetch()} />
            )}
            {loading ? (
              <p role="status" className="text-sm text-muted-foreground">
                {t("customerPortal.analytics.loading")}
              </p>
            ) : (
              <AppCustomerSpendTable
                app={app}
                rows={rows}
                monthly={monthlyTotals}
                selected={selectedTotals}
                totalSpend={totals.spend}
                startDate={startDate}
                endDate={endDate}
              />
            )}
          </section>
          <UsageBreakdownChart results={data.results} loading={loading} />
        </>
      )}
    </>
  );
}
