import { useTranslation } from "react-i18next";
import { useMemo, type ReactNode } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { ChartNoAxesCombined, ClipboardList, Layers, Server, Shield, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { DataTable, DataTableToolbar, useUrlTableState } from "@/components/shared/DataTable";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { controlPlanePages } from "@/lib/controlPlaneNavigation";

const sectionIcons = {
  "fleet-models": Layers,
  "fleet-teams": Users,
  "fleet-usage": ChartNoAxesCombined,
  dataplanes: Server,
  roles: Shield,
  audit: ClipboardList,
};

export function ControlPlaneSectionHeader({
  section,
  action,
}: {
  section: keyof typeof sectionIcons;
  action?: ReactNode;
}) {
  const { t } = useTranslation();
  const Icon = sectionIcons[section];
  const { label, description } = controlPlanePages[section];
  return (
    <PageHeader>
      <PageHeaderTitle>
        <Icon />
        {t(label)}
      </PageHeaderTitle>
      <PageHeaderDescription>{t(description)}</PageHeaderDescription>
      {action && <PageHeaderControls>{action}</PageHeaderControls>}
    </PageHeader>
  );
}

export function ErrorNotice({ error }: { error: unknown }) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  if (!error) return null;
  return (
    <Alert role="alert" variant="destructive">
      <AlertDescription>{error instanceof Error ? error.message : t("requestFailed")}</AlertDescription>
    </Alert>
  );
}

export function ScopedButton({
  reason,
  children,
  ...props
}: React.ComponentProps<typeof Button> & { reason?: string }) {
  if (!reason) return <Button {...props}>{children}</Button>;
  return (
    <Tooltip>
      <TooltipTrigger render={<span tabIndex={0} className="inline-flex cursor-not-allowed rounded-md" />}>
        <button
          type="button"
          disabled
          aria-disabled="true"
          className="pointer-events-none w-full rounded-md border px-3 py-2 text-sm text-muted-foreground"
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent role="tooltip">{reason}</TooltipContent>
    </Tooltip>
  );
}

export function ControlPlaneActionTooltip({ reason, children }: { reason?: string; children: ReactNode }) {
  if (!reason) return children;
  return (
    <Tooltip>
      <TooltipTrigger render={<span tabIndex={0} className="inline-flex cursor-not-allowed" />}>
        {children}
      </TooltipTrigger>
      <TooltipContent role="tooltip">{reason}</TooltipContent>
    </Tooltip>
  );
}

type ResultRow = { id: string; cells: readonly ReactNode[] };
const tableStateOptions = {
  sortFields: [],
  defaultSort: { id: "", desc: false },
  defaultPageSize: 25,
  filterColumns: [],
  keyPrefix: "cp",
};

export function ResultsTable({
  columns,
  rows,
  rowKeys,
  onRefresh,
  isFetching,
}: {
  columns: readonly string[];
  rows: readonly (readonly ReactNode[])[];
  rowKeys?: readonly (string | undefined)[];
  onRefresh?: () => void;
  isFetching?: boolean;
}) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const { search, setSearch, pagination, onPaginationChange } = useUrlTableState(tableStateOptions);
  const query = search.trim().toLocaleLowerCase();
  const data = useMemo(
    () =>
      rows
        .map((cells, index) => ({ id: rowKeys?.[index] ?? String(index), cells }))
        .filter(
          ({ cells }) =>
            !query ||
            cells.some(
              (cell) =>
                (typeof cell === "string" || typeof cell === "number") &&
                String(cell).toLocaleLowerCase().includes(query),
            ),
        ),
    [query, rowKeys, rows],
  );
  const definitions = useMemo<ColumnDef<ResultRow>[]>(
    () =>
      columns.map((title, index) => ({
        id: String(index),
        header: title,
        meta: { title },
        cell: ({ row }) => row.original.cells[index] ?? t("notSet"),
      })),
    [columns, t],
  );
  return (
    <DataTable
      data={data}
      columns={definitions}
      getRowId={(row) => row.id}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      toolbar={(table) => (
        <DataTableToolbar
          table={table}
          searchValue={search}
          onSearchChange={setSearch}
          onRefresh={onRefresh}
          isRefreshing={isFetching}
        />
      )}
    />
  );
}
