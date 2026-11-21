import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { customerBudgetPeriod } from "../../customers/_components/customerBudgetUsage";
import { Button } from "@/components/ui/button";
import { extractProxyErrorMessage } from "@/lib/http/client";
import {
  customerAppSpendCall,
  organizationActivityTotals,
  type CustomerUsageScope,
} from "../../customers/_components/customerUsageApi";
import { calendarDate, customerPresetRange } from "../../customers/_components/customerUsageRange";
import {
  CustomerSpendCells,
  CustomerSpendMetricsCells,
  CustomerSpendHeaders,
  CustomerSpendMetricsHeaders,
} from "../../customers/_components/CustomerSpendCells";
import CustomerTokenTotal from "../../customers/_components/CustomerTokenTotal";
import { appBudgetWindows, appUsageEntity } from "./appBudgetPresentation";
import type { App, AppMember, AppOrganization } from "./appForms";
import AppBudgetLimits from "./AppBudgetLimits";

export interface AppUsageContext {
  scope: CustomerUsageScope;
  startDate: string;
  endDate: string;
  organizationSpend: number | null;
}

export function AppBudgetRow({
  scope,
  app,
  row,
  sourceLabel,
  metrics,
  onAppSelect,
  onEdit,
}: {
  scope: "organization" | "member";
  app: App;
  row?: AppOrganization | AppMember;
  sourceLabel: string;
  metrics?: { layout: "full" | "compact"; context: AppUsageContext };
  onAppSelect?: (appId: string) => void;
  onEdit: () => void;
}) {
  const { t } = useTranslation();
  const organization = scope === "organization";
  const usage = metrics?.layout === "full";
  const showSourceColumn = organization || usage;
  const { effective_budget: budget, budget_usage: budgetUsage, spend } = row ?? {};
  const { primary } = appBudgetWindows(budget, budgetUsage);
  const period = customerBudgetPeriod(primary?.budget_duration ?? null);
  const periodLabel = period === "other" ? primary?.budget_duration : t(`apps.${period}`);
  return (
    <TableRow>
      <TableCell className="px-4 py-3">
        {onAppSelect ? (
          <button
            type="button"
            onClick={() => onAppSelect(app.app_id)}
            className="block text-left font-medium hover:text-primary hover:underline focus-visible:outline-2 focus-visible:outline-ring"
          >
            {app.alias}
          </button>
        ) : (
          <span className="block font-medium">{app.alias}</span>
        )}
        <span className="mt-1 block text-xs text-muted-foreground">
          {[showSourceColumn ? undefined : sourceLabel, periodLabel].filter(Boolean).join(" · ")}
        </span>
      </TableCell>
      {showSourceColumn && (
        <TableCell className="px-4 py-3 text-muted-foreground">
          <span className="block max-w-48 truncate" title={sourceLabel}>
            {sourceLabel}
          </span>
        </TableCell>
      )}
      {(!usage || !row) && (
        <TableCell className={organization ? "px-4 py-3" : "py-4"}>
          <AppBudgetLimits budget={budget} usage={budgetUsage} cumulativeSpend={spend} />
        </TableCell>
      )}
      {metrics && row && <AppBudgetSpendCells app={app} row={row} context={metrics.context} compact={!usage} />}
      <TableCell className={showSourceColumn ? "px-4 py-3 text-right" : undefined}>
        <Button
          type="button"
          variant="ghost"
          size={organization ? "sm" : "xs"}
          aria-label={t(organization ? "apps.editBudgetFor" : "apps.editMemberBudget", { alias: app.alias })}
          onClick={onEdit}
        >
          {t("apps.edit")}
        </Button>
      </TableCell>
    </TableRow>
  );
}

function AppBudgetSpendCells({
  app,
  row,
  context,
  compact = false,
}: {
  app: App;
  row: AppOrganization | AppMember;
  context: AppUsageContext;
  compact?: boolean;
}) {
  const { accessToken } = useAuthorized();
  const { app_id: appId, alias } = app;
  const month = customerPresetRange("month");
  const year = compact ? month : customerPresetRange("YTD");
  const periods = {
    monthStartDate: calendarDate(month.from),
    monthEndDate: calendarDate(month.to),
    yearStartDate: calendarDate(year.from),
    yearEndDate: calendarDate(year.to),
  };
  const activityOptions = {
    queryKey: ["customer-app-usage", appId, context.scope, context.startDate, context.endDate, periods],
    queryFn: ({ signal }: { signal: AbortSignal }) => {
      const request = { ...context, ...periods, appId, accessToken: accessToken!, signal };
      return customerAppSpendCall(request);
    },
    enabled: Boolean(accessToken),
  };
  const activity = useQuery(activityOptions);
  const organizationOptions = {
    queryKey: ["customer-app-organization-spend", context.scope.org, periods.monthStartDate, periods.monthEndDate],
    queryFn: ({ signal }: { signal: AbortSignal }) => {
      const request = {
        startDate: periods.monthStartDate,
        endDate: periods.monthEndDate,
        accessToken: accessToken!,
        signal,
      };
      return organizationActivityTotals(context.scope.org, request);
    },
    enabled: Boolean(accessToken) && context.organizationSpend === null,
  };
  const organization = useQuery(organizationOptions);
  const Cells = compact ? CustomerSpendMetricsCells : CustomerSpendCells;
  return (
    <Cells
      entity={appUsageEntity(appId, alias, row)}
      monthlySpend={activity.data?.monthlySpend ?? null}
      yearlySpend={activity.data?.yearlySpend ?? null}
      tokens={activity.data?.tokenDetails.tokens ?? null}
      tokenContent={
        activity.isError ? (
          <AppUsageRetry error={activity.error} onRetry={() => void activity.refetch()} />
        ) : (
          <CustomerTokenTotal usage={activity.data?.tokenDetails ?? null} />
        )
      }
      shareContent={
        context.organizationSpend === null && organization.isError ? (
          <AppUsageRetry error={organization.error} onRetry={() => void organization.refetch()} />
        ) : undefined
      }
      organizationSpend={context.organizationSpend ?? organization.data?.metadata.total_spend ?? null}
      usageLayout
    />
  );
}

function AppUsageRetry({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  const { t } = useTranslation();
  return (
    <Button variant="ghost" size="xs" title={extractProxyErrorMessage(error)} onClick={onRetry}>
      {t("customerPortal.analytics.unknownSpend")} · {t("apps.retry")}
    </Button>
  );
}

export function AppBudgetTable({
  scope,
  usageLayout = false,
  children,
}: {
  scope: "organization" | "member";
  usageLayout?: boolean;
  children: ReactNode;
}) {
  const { t } = useTranslation();
  const organization = scope === "organization";
  return (
    <div className={usageLayout ? "overflow-hidden rounded-lg border" : "overflow-x-auto"}>
      <Table className="w-full text-sm" aria-label={t(organization ? "apps.organizationTitle" : "apps.memberTitle")}>
        <TableHeader>
          <TableRow>
            <TableHead className="px-4 py-3">{t("apps.alias")}</TableHead>
            {(organization || usageLayout) && (
              <TableHead className="px-4">{t(organization ? "apps.profile" : "apps.source")}</TableHead>
            )}
            {usageLayout ? (
              <CustomerSpendHeaders usageLayout>
                <TableHead className="px-4">{t("customerPortal.hierarchy.maxBudget")}</TableHead>
              </CustomerSpendHeaders>
            ) : (
              <>
                <TableHead className="px-4">{t(organization ? "apps.effectiveBudget" : "apps.memberBudget")}</TableHead>
                {organization && <CustomerSpendMetricsHeaders usageLayout />}
              </>
            )}
            <TableHead className="px-4 text-right">{t("apps.actions")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>{children}</TableBody>
      </Table>
    </div>
  );
}
