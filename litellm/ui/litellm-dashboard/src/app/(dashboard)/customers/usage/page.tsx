"use client";

import { useTranslation } from "react-i18next";
import UsageStatStrip from "@/app/(dashboard)/usage/_components/components/overview/UsageStatStrip";
import { ErrorState } from "@/components/shared/ErrorState";
import { Button } from "@/components/ui/button";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import CustomerMemberSpend from "../_components/CustomerMemberSpend";
import CustomerUsageHeader from "../_components/CustomerUsageHeader";
import CustomerHierarchyNavigation, { customerHierarchyHref } from "../_components/CustomerHierarchyNavigation";
import CustomerBudgetSummary from "../_components/CustomerBudgetSummary";
import CustomerUsageRankings from "../_components/CustomerUsageRankings";
import CustomerUsageTrend from "../_components/CustomerUsageTrend";
import useCustomerHierarchyUsage from "../_components/useCustomerHierarchyUsage";

const METRIC_LABELS = {
  spend: "customerPortal.analytics.spend",
  tokens: "customerPortal.analytics.tokens",
  requests: "customerPortal.analytics.requests",
} as const;
const TOP_DIMENSION_LABELS = {
  models: "customerPortal.analytics.topModels",
  api_keys: "customerPortal.analytics.topKeys",
  teams: "customerPortal.analytics.topTeams",
} as const;

export default function CustomerHierarchyUsagePage() {
  const { t } = useTranslation();
  const {
    params,
    scope,
    dateValue,
    preset,
    usage,
    data,
    results,
    totals,
    modelColors,
    rows,
    drillDown,
    setDateRange,
    setDimension,
    selectOrganization,
    selectTeam,
    setMetric,
    setBucket,
    setSearch,
    setPage,
  } = useCustomerHierarchyUsage();
  const metricLabel = t(METRIC_LABELS[params.metric]);
  const budgetEntity = data?.member ?? data?.team ?? data?.organization;
  const budgetSettingsHref = scope ? `${customerHierarchyHref("manage", scope)}&tab=settings` : null;

  if (!scope)
    return (
      <div className="px-8 py-12">
        <h1 className="text-xl font-semibold">{t("customerPortal.analytics.title")}</h1>
        <p className="mt-2 text-sm text-muted-foreground">{t("customerPortal.analytics.selectOrganization")}</p>
        <a className="mt-5 inline-block text-sm font-medium text-primary hover:underline" href={uiHref("customers")}>
          {t("customerPortal.analytics.backToCustomers")}
        </a>
      </div>
    );

  return (
    <main
      className="mx-auto w-full max-w-[1600px] space-y-7 px-5 py-6 sm:px-8"
      aria-label={t("customerPortal.analytics.title")}
    >
      <CustomerUsageHeader
        scope={scope}
        data={data}
        dateValue={dateValue}
        preset={preset}
        onOrganizationClick={selectOrganization}
        onTeamClick={selectTeam}
        onDateRangeChange={setDateRange}
      />
      <CustomerHierarchyNavigation scope={scope} active="usage" />
      {usage.isError ? (
        <ErrorState
          title={t("customerPortal.analytics.error")}
          message={extractProxyErrorMessage(usage.error)}
          onRetry={() => void usage.refetch()}
        />
      ) : (
        <>
          <UsageStatStrip
            results={results}
            totals={totals}
            loading={usage.isPending}
            requestCountsPending={usage.isPending}
            budget={null}
            showBudgetHint={false}
            spendDetails={
              budgetEntity &&
              budgetSettingsHref && <CustomerBudgetSummary entity={budgetEntity} settingsHref={budgetSettingsHref} />
            }
          />
          <CustomerUsageTrend
            data={data}
            dimension={params.breakdown}
            onDimensionChange={setDimension}
            totals={totals}
            metric={params.metric}
            bucket={params.bucket}
            loading={usage.isPending}
            colorByModel={modelColors}
            onMetricChange={setMetric}
            onBucketChange={setBucket}
          />
          {data && <CustomerMemberSpend data={data} onMemberClick={(id) => drillDown("members", id)} />}
          {!usage.isPending && (
            <>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="text-base font-medium tracking-tight">{t(TOP_DIMENSION_LABELS[params.breakdown])}</h2>
                <p className="text-xs text-muted-foreground">
                  {t("customerPortal.analytics.sortedBy", { metric: metricLabel.toLocaleLowerCase() })}
                </p>
              </div>
              <CustomerUsageRankings
                rows={rows}
                dimension={params.breakdown}
                metric={params.metric}
                search={params.search}
                page={params.usagePage}
                colorByModel={modelColors}
                onSearchChange={setSearch}
                onPageChange={setPage}
                onDrillDown={drillDown}
              />
            </>
          )}
          {(params.team || params.member) && (
            <Button variant="link" className="h-auto p-0 text-xs text-muted-foreground" onClick={selectOrganization}>
              {t("customerPortal.analytics.allOrganizationUsage")}
            </Button>
          )}
          <p className="text-[11px] leading-5 text-muted-foreground">{t("customerPortal.analytics.freshness")}</p>
        </>
      )}
    </main>
  );
}
