import { useMemo } from "react";
import { skipToken, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { createSerializer } from "nuqs";
import { TableCell, TableHead, TableRow } from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { ErrorState } from "@/components/shared/ErrorState";
import { DataTablePagination, useUrlTableState } from "@/components/shared/DataTable";
import { uiHref } from "@/utils/uiHref";
import { extractProxyErrorMessage } from "@/lib/http/client";
import useAuthorized from "../../hooks/useAuthorized";
import { CustomerSpendCells, CustomerSpendTier } from "../../customers/_components/CustomerSpendCells";
import {
  CustomerUsageEditButton,
  CustomerUsageTable,
  CustomerUsageTableHeaders,
} from "../../customers/_components/CustomerUsageTable";
import CustomerTokenTotal from "../../customers/_components/CustomerTokenTotal";
import { customerAppsUsageCall } from "../../customers/_components/customerUsageApi";
import { customerAppParsers } from "../../customers/_components/CustomerHierarchyNavigation";
import { calendarDate, customerPresetRange } from "../../customers/_components/customerUsageRange";
import { useOverallAppBudgetGroups } from "./useAppBudgetGroups";
import { APP_RESOURCE_LABELS } from "./appForms";
import { applicationInitials, type ApplicationRow } from "./applications";
import { APPLICATION_COLUMNS, APPLICATION_TABLE, applicationTableRows, sortApplications } from "./applicationTableData";

const usageHref = createSerializer(customerAppParsers);

export default function ApplicationsTable({
  rows,
  loading,
  onOpen,
}: {
  rows: readonly ApplicationRow[];
  loading: boolean;
  onOpen?: (resourceId: string) => void;
}) {
  const { t } = useTranslation();
  const { search, setSearch, sorting, onSortingChange, pagination, onPaginationChange } =
    useUrlTableState(APPLICATION_TABLE);
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
  const tableRows = useMemo(
    () =>
      applicationTableRows(
        rows,
        onOpen && groups.isSuccess ? groups.data.default_budgets : undefined,
        usage.data,
        t("customerPortal.organizationGroups.customBadge"),
      ),
    [rows, onOpen, groups.isSuccess, groups.data, usage.data, t],
  );
  const sortedRows = useMemo(() => sortApplications(tableRows, search, sorting), [tableRows, search, sorting]);
  const { pageSize, pageIndex } = pagination;
  const currentPage = Math.min(pageIndex, Math.max(0, Math.ceil(sortedRows.length / pageSize) - 1));
  const visibleRows = sortedRows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
  const columnCount = APPLICATION_COLUMNS.length + Number(Boolean(onOpen));
  const emptyMessage =
    sortedRows.length === 0 ? t(search.trim() ? "apps.list.noMatches" : "apps.noResources") : undefined;
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b p-3">
        <Input
          className="max-w-xs"
          aria-label={t("apps.list.searchPlaceholder")}
          placeholder={t("apps.list.searchPlaceholder")}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <span className="text-xs text-muted-foreground">{t("apps.list.count", { count: sortedRows.length })}</span>
      </div>
      {usage.isError && (
        <ErrorState message={extractProxyErrorMessage(usage.error)} onRetry={() => void usage.refetch()} />
      )}
      <CustomerUsageTable
        label={t("apps.title")}
        columnCount={columnCount}
        message={loading ? t("apps.list.loading") : emptyMessage}
        headers={
          <>
            <CustomerUsageTableHeaders columns={APPLICATION_COLUMNS} sorting={sorting} onSort={onSortingChange} />
            {onOpen && <TableHead className="px-4 text-right">{t("apps.actions")}</TableHead>}
          </>
        }
      >
        {!loading &&
          visibleRows.map(({ resource, app, entity, total }) => (
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
                    {app ? (
                      <a
                        className="block max-w-52 truncate text-left font-medium hover:text-primary hover:underline"
                        title={resource.alias}
                        href={usageHref(uiHref("apps/usage"), { app_id: app.app_id })}
                      >
                        {resource.alias}
                      </a>
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
                  <CustomerUsageEditButton
                    aria-label={`${t("customerPortal.common.edit")} ${resource.alias}`}
                    onClick={() => onOpen(resource.resource_id)}
                  >
                    {t("customerPortal.common.edit")}
                  </CustomerUsageEditButton>
                </TableCell>
              )}
            </TableRow>
          ))}
      </CustomerUsageTable>
      <DataTablePagination
        page={currentPage}
        pageSize={pageSize}
        rowCount={sortedRows.length}
        onPageChange={(page) => onPaginationChange({ pageIndex: page, pageSize })}
        onPageSizeChange={(size) => onPaginationChange({ pageIndex: 0, pageSize: size })}
        isLoading={loading}
      />
    </>
  );
}
