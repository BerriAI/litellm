import { Pencil } from "lucide-react";
import { useMemo } from "react";
import { skipToken, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/shared/ErrorState";
import { cn } from "@/lib/cva.config";
import { extractProxyErrorMessage } from "@/lib/http/client";
import useAuthorized from "../../hooks/useAuthorized";
import { CustomerSpendCells, CustomerSpendTier } from "../../customers/_components/CustomerSpendCells";
import CustomerTokenTotal from "../../customers/_components/CustomerTokenTotal";
import { customerAppsUsageCall } from "../../customers/_components/customerUsageApi";
import { calendarDate, customerPresetRange } from "../../customers/_components/customerUsageRange";
import { appUsageEntity } from "./appBudgetPresentation";
import { useOverallAppBudgetGroups } from "./useAppBudgetGroups";
import { APP_RESOURCE_LABELS } from "./appForms";
import { applicationInitials, type ApplicationRow } from "./applications";

const COLUMNS = [
  { id: "application", label: "apps.list.columns.application" },
  { id: "group", label: "customerPortal.list.columns.tier" },
  { id: "spend", label: "customerPortal.list.columns.monthlySpend" },
  { id: "budget", label: "customerPortal.hierarchy.maxBudget" },
  { id: "share", label: "customerPortal.list.columns.overallSpendShare" },
  { id: "tokens", label: "customerPortal.list.columns.tokens" },
  { id: "status", label: "customerPortal.list.columns.status" },
] as const;

export default function ApplicationsTable({
  rows,
  loading,
  searching,
  onOpen,
}: {
  rows: readonly ApplicationRow[];
  loading: boolean;
  searching: boolean;
  onOpen?: (resourceId: string) => void;
}) {
  const { t } = useTranslation();
  const { accessToken, userId } = useAuthorized();
  const groups = useOverallAppBudgetGroups(Boolean(onOpen));
  const month = customerPresetRange("month");
  const startDate = calendarDate(month.from);
  const endDate = calendarDate(month.to);
  const appIds = useMemo(() => rows.flatMap(({ app }) => (app ? [app.app_id] : [])).sort(), [rows]);
  const usage = useQuery({
    queryKey: ["app-list-usage", userId, appIds, startDate, endDate, Boolean(onOpen)],
    queryFn:
      accessToken && appIds.length
        ? ({ signal }) => {
            const request = { accessToken, startDate, endDate, signal };
            return customerAppsUsageCall(appIds, Boolean(onOpen), request);
          }
        : skipToken,
  });
  const columnCount = COLUMNS.length + Number(Boolean(onOpen));
  return (
    <>
      {usage.isError && (
        <ErrorState message={extractProxyErrorMessage(usage.error)} onRetry={() => void usage.refetch()} />
      )}
      <Table aria-label={t("apps.title")}>
        <TableHeader>
          <TableRow>
            {COLUMNS.map(({ id, label }) => (
              <TableHead
                key={id}
                className={cn(
                  "px-4",
                  id === "spend" && "w-[30%] min-w-80",
                  (id === "status" || id === "tokens") && "text-right",
                )}
              >
                {t(label)}
              </TableHead>
            ))}
            {onOpen && <TableHead className="px-4 text-right">{t("apps.actions")}</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading && (
            <TableRow>
              <TableCell colSpan={columnCount} className="py-8 text-center text-muted-foreground">
                {t("apps.list.loading")}
              </TableCell>
            </TableRow>
          )}
          {!loading && rows.length === 0 && (
            <TableRow>
              <TableCell colSpan={columnCount} className="py-8 text-center text-muted-foreground">
                {t(searching ? "apps.list.noMatches" : "apps.noResources")}
              </TableCell>
            </TableRow>
          )}
          {!loading &&
            rows.map(({ resource, app }) => {
              const entity = {
                ...appUsageEntity(resource.resource_id, resource.alias, {
                  spend: app?.spend ?? 0,
                  effective_budget: app?.budget,
                  budget_usage: app?.budget_usage,
                }),
                budgetGroup:
                  onOpen && groups.isSuccess
                    ? groups.data.default_budgets.find((group) => group.budget_id === app?.budget_id)?.name
                    : "—",
              };
              const total = app ? usage.data?.get(app.app_id) : null;
              return (
                <TableRow key={resource.resource_id}>
                  <TableCell className="px-4 py-5">
                    <div className="flex items-center gap-3">
                      <span
                        aria-hidden="true"
                        className="flex size-9 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium"
                      >
                        {applicationInitials(resource.alias)}
                      </span>
                      <div className="min-w-0">
                        {onOpen ? (
                          <button
                            type="button"
                            className="block max-w-52 truncate text-left font-medium hover:text-primary hover:underline"
                            title={resource.alias}
                            onClick={() => onOpen(resource.resource_id)}
                          >
                            {resource.alias}
                          </button>
                        ) : (
                          <p className="max-w-52 truncate font-medium" title={resource.alias}>
                            {resource.alias}
                          </p>
                        )}
                        <p
                          className="max-w-52 truncate font-mono text-[11px] text-muted-foreground"
                          title={resource.resource_id}
                        >
                          {resource.resource_id}
                        </p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {t(APP_RESOURCE_LABELS[resource.resource_type])}
                        </p>
                      </div>
                    </div>
                  </TableCell>
                  <CustomerSpendTier entity={entity} />
                  <CustomerSpendCells
                    entity={entity}
                    monthlySpend={total?.spend ?? null}
                    yearlySpend={null}
                    tokens={total?.tokens ?? null}
                    tokenContent={<CustomerTokenTotal usage={total ?? null} />}
                    organizationSpend={usage.data?.get(null)?.spend ?? null}
                    shareLabel={t("customerPortal.list.columns.overallSpendShare")}
                    usageLayout
                    showYearly={false}
                  />
                  {onOpen && (
                    <TableCell className="px-4 text-right">
                      <Button
                        variant="outline"
                        size="sm"
                        aria-label={`${t("customerPortal.common.edit")} ${resource.alias}`}
                        onClick={() => onOpen(resource.resource_id)}
                      >
                        <Pencil aria-hidden="true" className="size-3.5" />
                        {t("customerPortal.common.edit")}
                      </Button>
                    </TableCell>
                  )}
                </TableRow>
              );
            })}
        </TableBody>
      </Table>
    </>
  );
}
