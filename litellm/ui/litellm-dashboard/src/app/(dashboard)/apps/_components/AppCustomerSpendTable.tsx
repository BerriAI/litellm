"use client";

import { useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { TableCell, TableHead, TableRow } from "@/components/ui/table";
import { DataTablePagination, useUrlTableState } from "@/components/shared/DataTable";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { isProxyAdminRole } from "@/utils/roles";
import useAuthorized from "../../hooks/useAuthorized";
import {
  CustomerSpendCells,
  CustomerSpendHeaders,
  CustomerSpendIdentity,
  CustomerSpendTier,
} from "../../customers/_components/CustomerSpendCells";
import { CustomerUsageEditButton, CustomerUsageTable } from "../../customers/_components/CustomerUsageTable";
import CustomerTokenTotal from "../../customers/_components/CustomerTokenTotal";
import { customerHierarchyHref } from "../../customers/_components/CustomerHierarchyNavigation";
import type { CustomerUsageRow } from "../../customers/_components/customerUsageData";
import type { CustomerUsageTotals } from "../../customers/_components/customerUsageTotals";
import { AppOrganizationBudgetDialog, appOrganizationSourceLabel } from "./AppOrganizationBudgetDialog";
import { appUsageEntity } from "./appBudgetPresentation";
import { appOrganizationPresetId, type App } from "./appForms";
import { loadOrganizationApp, organizationAppKey } from "./appApi";
import { useAppBudgetGroups } from "./useAppBudgetGroups";

const TABLE_OPTIONS = {
  sortFields: ["spend"],
  defaultSort: { id: "spend", desc: true },
  defaultPageSize: 5,
  filterColumns: [],
  urlKeys: { page: "usagePage" },
};

export default function AppCustomerSpendTable({
  app,
  rows,
  monthly,
  selected,
  totalSpend,
  startDate,
  endDate,
}: {
  app: App;
  rows: readonly CustomerUsageRow[];
  monthly?: CustomerUsageTotals;
  selected: CustomerUsageTotals;
  totalSpend: number;
  startDate: string;
  endDate: string;
}) {
  const { t } = useTranslation();
  const { userRole, isViewOnly } = useAuthorized();
  const isAdmin = isProxyAdminRole(userRole);
  const canEdit = isAdmin && !isViewOnly;
  const [editing, setEditing] = useState<CustomerUsageRow | null>(null);
  const { search, setSearch, pagination, onPaginationChange } = useUrlTableState(TABLE_OPTIONS);
  const filtered = rows.filter((row) =>
    `${row.label} ${row.id}`.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()),
  );
  const { pageSize, pageIndex } = pagination;
  const page = Math.min(pageIndex, Math.max(0, Math.ceil(filtered.length / pageSize) - 1));
  const visible = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const associations = useQueries({
    queries: visible.map((row) => ({
      queryKey: organizationAppKey(app.app_id, row.id),
      queryFn: () => loadOrganizationApp(app.app_id, row.id),
      enabled: Boolean(row.id),
      retry: false,
    })),
  });
  const groups = useAppBudgetGroups(
    visible.map((row, index) =>
      row.id && !associations[index]?.data?.budget ? appOrganizationPresetId(associations[index]?.data, app) : null,
    ),
    isAdmin,
  );
  const shareLabel = t("apps.usage.spendShare");
  const tableRows = visible.map((customer, index) => {
    const association = associations[index];
    const row = association.data ?? null;
    const inherited = groups.displayBudgets.find((group) => group.budget_id === appOrganizationPresetId(row, app));
    const needsInherited = Boolean(customer.id && !row && app.default_tier_budget_id);
    const missingInherited = needsInherited && !inherited;
    const error = association.error ?? (needsInherited ? groups.error : null);
    const budgetUnknown = association.isPending || error || missingInherited;
    const unavailable = Boolean(customer.id && budgetUnknown);
    const loading = association.isPending || (isAdmin && groups.isPending);
    const entity = {
      ...appUsageEntity(
        customer.id,
        customer.label,
        row ?? { spend: 0, effective_budget: customer.id ? inherited : null },
      ),
      budgetGroup: customer.id ? appOrganizationSourceLabel(app, row, groups.displayBudgets, t) : "—",
    };
    return { customer, association, error, unavailable, loading, entity };
  });
  return (
    <>
      <p className="text-sm text-muted-foreground">{t("apps.usage.budgetHint")}</p>
      <div className="overflow-hidden rounded-lg border">
        <div className="border-b p-3">
          <Input
            className="max-w-xs"
            aria-label={t("customerPortal.list.searchPlaceholder")}
            placeholder={t("customerPortal.list.searchPlaceholder")}
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        <CustomerUsageTable
          label={t("customerPortal.list.title")}
          columnCount={7 + Number(canEdit)}
          message={visible.length === 0 ? t("customerPortal.analytics.empty") : undefined}
          headers={
            <>
              <TableHead className="px-4">{t("customerPortal.list.columns.customer")}</TableHead>
              <TableHead className="px-4">{t("apps.profile")}</TableHead>
              <CustomerSpendHeaders usageLayout showYearly={false} shareLabel={shareLabel}>
                <TableHead className="px-4">{t("customerPortal.hierarchy.maxBudget")}</TableHead>
              </CustomerSpendHeaders>
              {canEdit && <TableHead className="px-4 text-right">{t("apps.actions")}</TableHead>}
            </>
          }
        >
          {tableRows.map(({ customer, association, error, unavailable, loading, entity }) => {
            return (
              <TableRow key={customer.id}>
                <TableCell className="px-4 py-5">
                  {isAdmin && customer.id ? (
                    <CustomerSpendIdentity
                      entity={entity}
                      usageLayout
                      href={customerHierarchyHref(
                        "usage",
                        { org: customer.id, team: null, member: null },
                        { app_id: app.app_id, from: startDate, to: endDate },
                      )}
                    />
                  ) : (
                    <span className="font-medium">{customer.label}</span>
                  )}
                </TableCell>
                <CustomerSpendTier
                  entity={unavailable ? { ...entity, budgetGroup: t("customerPortal.analytics.unknownSpend") } : entity}
                />
                <CustomerSpendCells
                  entity={entity}
                  monthlySpend={monthly ? monthly.get(customer.id)?.spend ?? 0 : null}
                  yearlySpend={null}
                  tokens={customer.tokens}
                  tokenContent={<CustomerTokenTotal usage={selected.get(customer.id) ?? null} />}
                  shareSpend={customer.spend}
                  organizationSpend={totalSpend}
                  shareLabel={shareLabel}
                  usageLayout
                  showYearly={false}
                  budgetUnavailable={
                    unavailable &&
                    (error ? (
                      <ErrorState
                        message={extractProxyErrorMessage(error)}
                        onRetry={() =>
                          void Promise.all([association.refetch(), ...(isAdmin ? [groups.refetch()] : [])])
                        }
                      />
                    ) : (
                      <span className="text-sm text-muted-foreground">
                        {t(loading ? "apps.loading" : "customerPortal.analytics.budgetUsageUnavailable")}
                      </span>
                    ))
                  }
                />
                {canEdit && (
                  <TableCell className="px-4 text-right">
                    {customer.id && (
                      <CustomerUsageEditButton
                        aria-label={t("apps.editBudgetFor", { alias: customer.label })}
                        onClick={() => setEditing(customer)}
                      >
                        {t("apps.edit")}
                      </CustomerUsageEditButton>
                    )}
                  </TableCell>
                )}
              </TableRow>
            );
          })}
        </CustomerUsageTable>
        <DataTablePagination
          page={page}
          pageSize={pageSize}
          pageSizeOptions={[5, 10, 25]}
          rowCount={filtered.length}
          onPageChange={(pageIndex) => onPaginationChange({ pageIndex, pageSize })}
          onPageSizeChange={(size) => onPaginationChange({ pageIndex: 0, pageSize: size })}
        />
      </div>
      {editing && (
        <AppOrganizationBudgetDialog
          app={app}
          organizationId={editing.id}
          organizationAlias={editing.label}
          onClose={() => setEditing(null)}
        />
      )}
    </>
  );
}
