import { skipToken, useQuery } from "@tanstack/react-query";
import { createSerializer, parseAsString, parseAsStringLiteral, useQueryState, useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import {
  CustomerDetailHeader,
  CustomerDetailLayout,
  CustomerDetailNavigation,
} from "../../customers/_components/CustomerDetailLayout";
import { CustomerUsageDatePicker, CustomerUsageStats } from "../../customers/_components/CustomerUsageSummary";
import { CustomerBudgetSummaryContent } from "../../customers/_components/CustomerBudgetSummary";
import CustomerUsageAnalysis from "../../customers/_components/CustomerUsageAnalysis";
import CustomerUsageAppFilter from "../../customers/_components/CustomerUsageAppFilter";
import { customerAppParsers } from "../../customers/_components/CustomerHierarchyNavigation";
import { customerAppGroupUsageCall } from "../../customers/_components/customerUsageApi";
import {
  calendarDate,
  customerUsageDateParsers,
  useCustomerUsageRange,
} from "../../customers/_components/customerUsageRange";
import { overviewTotals } from "../../usage/_components/components/overview/overviewData";
import useAuthorized from "../../hooks/useAuthorized";
import AppGroupCustomers from "./AppGroupCustomers";
import ApplicationsTable from "./ApplicationsTable";
import AppSettingsDialog from "./AppSettingsDialog";
import { appUsageEntity } from "./appBudgetPresentation";
import { loadApps } from "./appApi";
import { grantedApplicationRows } from "./applications";
import type { AppGroup } from "./appGroupApi";
import type { GroupBudgetTarget } from "./AppGroupBudgetDialog";

const GROUP_TABS = [
  ["usage", "customerPortal.hierarchy.usage"],
  ["apps", "apps.groups.apps"],
  ["customers", "apps.groups.customers"],
] as const;
export const appGroupDetailParsers = {
  ...customerAppParsers,
  app_group: parseAsString,
  app: parseAsString,
  group_tab: parseAsStringLiteral(GROUP_TABS.map(([tab]) => tab)).withDefault("usage"),
};
const detailParsers = { ...appGroupDetailParsers, ...customerUsageDateParsers };
const detailHref = createSerializer(detailParsers);
const GROUP_APPS_TABLE_PREFIX = "group_apps_";

export default function AppGroupDetails({
  group,
  onEdit,
  onBack,
}: {
  group: AppGroup;
  onEdit: (target: GroupBudgetTarget) => void;
  onBack: () => void;
}) {
  const { t } = useTranslation();
  const { accessToken, userId } = useAuthorized();
  const [params, setParams] = useQueryStates(detailParsers);
  const [selectedApp, setSelectedApp] = useQueryState("app", parseAsString);
  const { dateValue, onDateChange, preset } = useCustomerUsageRange();
  const startDate = calendarDate(dateValue.from);
  const endDate = calendarDate(dateValue.to);
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => loadApps() });
  const rows = grantedApplicationRows((apps.data ?? []).filter((app) => app.app_group_id === group.app_group_id));
  const resource = rows.find((row) => row.resource.resource_id === selectedApp)?.resource;
  const usage = useQuery({
    queryKey: ["app-groups", "usage", userId, group.app_group_id, params.app_id, startDate, endDate],
    queryFn: accessToken
      ? ({ signal }) => {
          const request = {
            accessToken,
            appGroupId: group.app_group_id,
            appId: params.app_id,
            startDate,
            endDate,
            signal,
          };
          return customerAppGroupUsageCall(request);
        }
      : skipToken,
    enabled: params.group_tab === "usage",
  });
  const data = usage.data;
  const totals = overviewTotals(data?.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA, null);
  const entity = appUsageEntity(group.app_group_id, group.name, {
    spend: group.spend,
    effective_budget: group.budget,
    budget_usage: group.budget_usage,
  });
  const editGroup = () => onEdit({ kind: "group", group });
  const applications = (
    <section className="space-y-4" aria-label={t("apps.groups.apps")}>
      <h2 className="text-base font-medium">{t("apps.groups.apps")}</h2>
      {apps.isError ? (
        <ErrorState message={extractProxyErrorMessage(apps.error)} onRetry={() => void apps.refetch()} />
      ) : (
        <ApplicationsTable
          rows={rows}
          loading={apps.isPending}
          keyPrefix={GROUP_APPS_TABLE_PREFIX}
          usageScope={{ appGroupId: group.app_group_id, startDate, endDate }}
          onOpen={(id) => void setSelectedApp(id)}
        />
      )}
    </section>
  );
  return (
    <CustomerDetailLayout label={t("apps.groups.title")}>
      <CustomerDetailHeader
        title={group.name}
        breadcrumbs={
          <button className="hover:text-foreground" onClick={onBack}>
            {t("apps.groups.title")}
          </button>
        }
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="outline" onClick={editGroup}>
              {t("apps.groups.edit")}
            </Button>
            <CustomerUsageDatePicker value={dateValue} presetShortLabel={preset} onValueChange={onDateChange} />
          </div>
        }
      >
        {group.description && <p className="mt-2 text-sm text-muted-foreground">{group.description}</p>}
        <p className="mt-2 text-sm text-muted-foreground">{t("apps.groups.usageHint")}</p>
      </CustomerDetailHeader>
      <CustomerDetailNavigation
        label={t("apps.groups.title")}
        items={GROUP_TABS.map(([tab, label]) => ({
          href: detailHref(uiHref("app-groups"), { ...params, group_tab: tab }),
          label: t(label),
          active: params.group_tab === tab,
        }))}
      />
      {params.group_tab === "usage" && (
        <>
          {usage.isError ? (
            <ErrorState
              title={t("customerPortal.analytics.error")}
              message={extractProxyErrorMessage(usage.error)}
              onRetry={() => void usage.refetch()}
            />
          ) : (
            <>
              <CustomerUsageStats
                results={data?.results ?? []}
                totals={totals}
                loading={usage.isPending}
                surcharge={data?.metadata.total_customer_margin_spend}
              >
                <CustomerBudgetSummaryContent entity={entity} onEdit={editGroup} />
              </CustomerUsageStats>
              {applications}
              <CustomerUsageAnalysis
                data={data}
                loading={usage.isPending}
                appFilter={
                  <CustomerUsageAppFilter
                    appGroupId={group.app_group_id}
                    value={params.app_id}
                    onChange={(app_id) => void setParams({ app_id })}
                  />
                }
              />
            </>
          )}
        </>
      )}
      {params.group_tab === "apps" && applications}
      {params.group_tab === "customers" && <AppGroupCustomers group={group} onEdit={onEdit} />}
      {resource && <AppSettingsDialog resource={resource} onClose={() => void setSelectedApp(null)} />}
    </CustomerDetailLayout>
  );
}
