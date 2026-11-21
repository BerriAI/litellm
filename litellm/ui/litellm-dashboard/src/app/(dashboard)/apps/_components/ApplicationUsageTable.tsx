import { useMemo, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { TableCell, TableHead, TableRow } from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { DataTablePagination, useUrlTableState } from "@/components/shared/DataTable";
import { CustomerSpendCells, CustomerSpendTier } from "../../customers/_components/CustomerSpendCells";
import {
  CustomerUsageEditButton,
  CustomerUsageTable,
  CustomerUsageTableHeaders,
} from "../../customers/_components/CustomerUsageTable";
import CustomerTokenTotal from "../../customers/_components/CustomerTokenTotal";
import { APP_RESOURCE_LABELS } from "./appForms";
import { applicationInitials } from "./applications";
import {
  APPLICATION_COLUMNS,
  APPLICATION_TABLE,
  sortApplications,
  type ApplicationUsageTableRow,
  type ApplicationSpendPeriod,
} from "./applicationTableData";

export default function ApplicationUsageTable({
  rows,
  loading,
  overallSpend,
  kind = "apps",
  onEdit,
  onOpen,
  actions,
  keyPrefix,
  spendPeriod = "monthly",
}: {
  rows: readonly ApplicationUsageTableRow[];
  loading: boolean;
  overallSpend: number | null;
  kind?: "apps" | "groups";
  onEdit?: (id: string) => void;
  onOpen?: (id: string) => void;
  actions?: ReactNode;
  keyPrefix?: string;
  spendPeriod?: ApplicationSpendPeriod;
}) {
  const { t } = useTranslation();
  const defaultPrefix = kind === "groups" ? "groups_" : "";
  const tableOptions = {
    ...APPLICATION_TABLE,
    defaultPageSize: kind === "groups" ? 10 : APPLICATION_TABLE.defaultPageSize,
    keyPrefix: keyPrefix ?? defaultPrefix,
  };
  const { search, setSearch, sorting, onSortingChange, pagination, onPaginationChange } =
    useUrlTableState(tableOptions);
  const columns = APPLICATION_COLUMNS.map((column) => {
    if (column.id === "application" && kind === "groups") return { ...column, label: "apps.groups.parent" as const };
    if (column.id === "spend" && spendPeriod === "selected")
      return { ...column, label: "customerPortal.list.columns.spend" as const };
    return column;
  });
  const sortedRows = useMemo(() => sortApplications(rows, search, sorting), [rows, search, sorting]);
  const { pageSize, pageIndex } = pagination;
  const currentPage = Math.min(pageIndex, Math.max(0, Math.ceil(sortedRows.length / pageSize) - 1));
  const visibleRows = sortedRows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
  const columnCount = columns.length + Number(Boolean(onEdit));
  const noMatches = kind === "groups" ? "apps.groups.noMatches" : "apps.list.noMatches";
  const noResources = kind === "groups" ? "apps.groups.empty" : "apps.noResources";
  const emptyMessage = sortedRows.length === 0 ? t(search.trim() ? noMatches : noResources) : undefined;
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b p-3">
        <Input
          className="max-w-xs"
          aria-label={t(kind === "groups" ? "apps.groups.search" : "apps.list.searchPlaceholder")}
          placeholder={t(kind === "groups" ? "apps.groups.search" : "apps.list.searchPlaceholder")}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        {actions}
        <span className="text-xs text-muted-foreground">
          {t(kind === "groups" ? "apps.groups.count" : "apps.list.count", { count: sortedRows.length })}
        </span>
      </div>
      <CustomerUsageTable
        label={t(kind === "groups" ? "apps.groups.title" : "apps.title")}
        columnCount={columnCount}
        message={loading ? t("apps.list.loading") : emptyMessage}
        headers={
          <>
            <CustomerUsageTableHeaders columns={columns} sorting={sorting} onSort={onSortingChange} />
            {onEdit && <TableHead className="px-4 text-right">{t("apps.actions")}</TableHead>}
          </>
        }
      >
        {!loading &&
          visibleRows.map(({ entity, total, href, description, resourceType }) => (
            <TableRow key={entity.id}>
              <TableCell className="px-4 py-5">
                <div className="flex items-center gap-3">
                  <span
                    aria-hidden="true"
                    className="flex size-9 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium"
                  >
                    {applicationInitials(entity.label)}
                  </span>
                  <div className="min-w-0">
                    <ApplicationName entity={entity} href={href} onOpen={onOpen} />
                    <p className="max-w-52 truncate font-mono text-[11px] text-muted-foreground" title={entity.id}>
                      {entity.id}
                    </p>
                    {resourceType && (
                      <p className="mt-1 text-xs text-muted-foreground">{t(APP_RESOURCE_LABELS[resourceType])}</p>
                    )}
                    {description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}
                  </div>
                </div>
              </TableCell>
              <CustomerSpendTier entity={entity} />
              <CustomerSpendCells
                entity={entity}
                monthlySpend={total?.spend ?? null}
                selectedSpend={spendPeriod === "selected" ? total?.spend ?? null : undefined}
                yearlySpend={null}
                tokens={total?.tokens ?? null}
                tokenContent={<CustomerTokenTotal usage={total ?? null} />}
                organizationSpend={overallSpend}
                shareLabel={t("customerPortal.list.columns.overallSpendShare")}
                usageLayout
                showYearly={false}
              />
              {onEdit && (
                <TableCell className="px-4 text-right">
                  <CustomerUsageEditButton
                    aria-label={`${t("customerPortal.common.edit")} ${entity.label}`}
                    onClick={() => onEdit(entity.id)}
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
        pageSizeOptions={kind === "groups" ? [10, 25, 50, 100] : undefined}
        rowCount={sortedRows.length}
        onPageChange={(page) => onPaginationChange({ pageIndex: page, pageSize })}
        onPageSizeChange={(size) => onPaginationChange({ pageIndex: 0, pageSize: size })}
        isLoading={loading}
      />
    </>
  );
}

function ApplicationName({
  entity,
  href,
  onOpen,
}: Pick<ApplicationUsageTableRow, "entity" | "href"> & { onOpen?: (id: string) => void }) {
  const className = "block max-w-52 truncate text-left font-medium hover:text-primary hover:underline";
  if (href)
    return (
      <a className={className} title={entity.label} href={href}>
        {entity.label}
      </a>
    );
  if (onOpen)
    return (
      <button className={className} title={entity.label} onClick={() => onOpen(entity.id)}>
        {entity.label}
      </button>
    );
  return (
    <p className="max-w-52 truncate font-medium" title={entity.label}>
      {entity.label}
    </p>
  );
}
