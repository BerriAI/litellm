"use client";

import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/shared/ErrorState";
import { Button } from "@/components/ui/button";
import { extractProxyErrorMessage } from "@/lib/http/client";
import CustomerMemberSpend from "../_components/CustomerMemberSpend";
import CustomerUsageHeader from "../_components/CustomerUsageHeader";
import { CustomerDetailLayout } from "../_components/CustomerDetailLayout";
import CustomerHierarchyNavigation, { useCustomerHierarchyLinks } from "../_components/CustomerHierarchyNavigation";
import CustomerBudgetSummary from "../_components/CustomerBudgetSummary";
import { CustomerUsageStats } from "../_components/CustomerUsageSummary";
import CustomerUsageAnalysis from "../_components/CustomerUsageAnalysis";
import useCustomerHierarchyUsage from "../_components/useCustomerHierarchyUsage";
import CustomerOrganizationApps from "../../apps/_components/CustomerOrganizationApps";
import CustomerMemberApps from "../../apps/_components/CustomerMemberApps";
import CustomerUsageAppFilter from "../_components/CustomerUsageAppFilter";
import { calendarDate } from "../_components/customerUsageRange";

export default function CustomerHierarchyUsagePage() {
  const { t } = useTranslation();
  const { customerHierarchyHref, customersHref } = useCustomerHierarchyLinks();
  const {
    params,
    scope,
    dateValue,
    preset,
    usage,
    data,
    results,
    totals,
    drillDown,
    setDateRange,
    selectOrganization,
    selectTeam,
    setApp,
    appOrganizationSpend,
  } = useCustomerHierarchyUsage();
  const budgetEntity = data?.member ?? data?.team ?? data?.organization;

  if (!scope)
    return (
      <div className="px-8 py-12">
        <h1 className="text-xl font-semibold">{t("customerPortal.analytics.title")}</h1>
        <p className="mt-2 text-sm text-muted-foreground">{t("customerPortal.analytics.selectOrganization")}</p>
        <a className="mt-5 inline-block text-sm font-medium text-primary hover:underline" href={customersHref}>
          {t("customerPortal.analytics.backToCustomers")}
        </a>
      </div>
    );

  const budgetSettingsHref = `${customerHierarchyHref("manage", scope)}&tab=settings`;
  const isOrganizationScope = !scope.team && !scope.member;
  const appUsageContext = {
    scope,
    startDate: calendarDate(dateValue.from),
    endDate: calendarDate(dateValue.to),
    organizationSpend: appOrganizationSpend,
  };

  return (
    <CustomerDetailLayout label={t("customerPortal.analytics.title")}>
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
          <CustomerUsageStats
            results={results}
            totals={totals}
            loading={usage.isPending}
            surcharge={data?.metadata.total_customer_margin_spend}
          >
            {budgetEntity && (
              <CustomerBudgetSummary entity={budgetEntity} settingsHref={budgetSettingsHref} scope={scope} />
            )}
          </CustomerUsageStats>
          {isOrganizationScope && (
            <CustomerOrganizationApps
              key={scope.org}
              organizationId={scope.org}
              organizationAlias={data?.organization.label}
              onAppSelect={setApp}
              usageContext={appUsageContext}
            />
          )}
          {scope.member && (
            <CustomerMemberApps
              organizationId={scope.org!}
              key={scope.member}
              memberId={scope.member}
              usageContext={appUsageContext}
              onAppSelect={setApp}
            />
          )}
          <CustomerUsageAnalysis
            data={data}
            loading={usage.isPending}
            onDrillDown={drillDown}
            appFilter={<CustomerUsageAppFilter value={params.app_id} onChange={setApp} />}
          >
            {data && <CustomerMemberSpend data={data} onMemberClick={(id) => drillDown("members", id)} />}
          </CustomerUsageAnalysis>
          {(params.team || params.member) && (
            <Button variant="link" className="h-auto p-0 text-xs text-muted-foreground" onClick={selectOrganization}>
              {t("customerPortal.analytics.allOrganizationUsage")}
            </Button>
          )}
        </>
      )}
    </CustomerDetailLayout>
  );
}
