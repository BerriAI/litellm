"use client";

import type { ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import { ScrollText } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { useUserEmailLookup } from "@/app/(dashboard)/hooks/users/useUsers";
import { DataTable, DataTableFilterDrawer, DataTableToolbar } from "@/components/shared/DataTable";

import type { Team } from "../../key_team_helpers/key_list";
import type { LogEntry } from "../types";
import { CREDENTIAL_LABELS, SPAN_TYPE_LABELS } from "../constants";
import { LOG_FILTER_IDS, LOG_FILTER_LABELS, type LogsWindow } from "./useLogFilterLogic";
import { RequestLogsFilters } from "./RequestLogsFilters";
import { getRequestLogsTableColumns } from "./RequestLogsTableColumns";

interface RequestLogsTableProps {
  data: LogEntry[];
  rowCount: number;
  isLoading: boolean;
  isError?: boolean;
  errorMessage?: ReactNode;
  isRefreshing: boolean;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  sorting: SortingState;
  onSortingChange: OnChangeFn<SortingState>;
  columnFilters: ColumnFiltersState;
  onColumnFiltersChange: OnChangeFn<ColumnFiltersState>;
  searchValue: string;
  onSearchChange: (value: string) => void;
  onRefresh: () => void;
  onRowClick: (log: LogEntry) => void;
  onKeyHashClick: (keyHash: string) => void;
  onSessionClick: (log: LogEntry) => void;
  teams: Team[];
  logsWindow: LogsWindow;
  toolbarChildren?: ReactNode;
}

const formatFilterValue = (columnId: string, value: unknown, t: TFunction): string => {
  if (columnId === LOG_FILTER_IDS.SPAN_TYPE) {
    return Object.hasOwn(SPAN_TYPE_LABELS, String(value)) ? t(SPAN_TYPE_LABELS[String(value)]) : String(value);
  }
  if (columnId === LOG_FILTER_IDS.CREDENTIAL) {
    return Object.hasOwn(CREDENTIAL_LABELS, String(value)) ? t(CREDENTIAL_LABELS[String(value)]) : String(value);
  }
  return Array.isArray(value) ? value.join(", ") : String(value);
};

function RequestLogsEmptyState({ filtered }: { filtered: boolean }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <ScrollText className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {filtered ? t("logs.request.table.emptyFilteredTitle") : t("logs.request.table.emptyTitle")}
      </div>
      <div className="max-w-xs text-center text-sm text-muted-foreground">
        {filtered ? t("logs.request.table.emptyFilteredDescription") : t("logs.request.table.emptyDescription")}
      </div>
    </div>
  );
}

export function RequestLogsTable({
  data,
  rowCount,
  isLoading,
  isError = false,
  errorMessage,
  isRefreshing,
  pagination,
  onPaginationChange,
  sorting,
  onSortingChange,
  columnFilters,
  onColumnFiltersChange,
  searchValue,
  onSearchChange,
  onRefresh,
  onRowClick,
  onKeyHashClick,
  onSessionClick,
  teams,
  logsWindow,
  toolbarChildren,
}: RequestLogsTableProps) {
  const { t } = useTranslation();
  const [filtersOpen, setFiltersOpen] = useState(false);

  const userIds = useMemo(() => data.flatMap((log) => (log.user ? [log.user] : [])), [data]);
  const { data: emailByUserId } = useUserEmailLookup(userIds);

  const columns = useMemo(() => {
    const resolveUserEmail = (userId: string) => emailByUserId?.[userId];
    return getRequestLogsTableColumns({ onKeyHashClick, onSessionClick, resolveUserEmail }, t);
  }, [onKeyHashClick, onSessionClick, emailByUserId, t]);

  const filterLabels = useMemo(
    () => Object.fromEntries(Object.entries(LOG_FILTER_LABELS).map(([id, labelKey]) => [id, t(labelKey)])),
    [t],
  );

  const isFiltered = columnFilters.length > 0 || searchValue !== "";

  return (
    <DataTable
      data={data}
      columns={columns}
      getRowId={(row) => row.request_id}
      fillHeight
      sortingMode="server"
      sorting={sorting}
      onSortingChange={onSortingChange}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={rowCount}
      filterMode="server"
      columnFilters={columnFilters}
      onColumnFiltersChange={onColumnFiltersChange}
      isLoading={isLoading}
      isError={isError}
      errorMessage={errorMessage}
      onRetry={onRefresh}
      loadingMessage={t("logs.request.table.loading")}
      noDataMessage={<RequestLogsEmptyState filtered={isFiltered} />}
      size="compact"
      onRowClick={onRowClick}
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={searchValue}
            onSearchChange={onSearchChange}
            searchPlaceholder={t("logs.request.table.searchPlaceholder")}
            onRefresh={onRefresh}
            isRefreshing={isRefreshing}
            onOpenFilters={() => setFiltersOpen(true)}
            filterLabels={filterLabels}
            formatFilterValue={(columnId, value) => formatFilterValue(columnId, value, t)}
            showViewOptions={false}
          >
            {toolbarChildren}
          </DataTableToolbar>
          <DataTableFilterDrawer
            table={table}
            open={filtersOpen}
            onOpenChange={setFiltersOpen}
            title={t("logs.request.table.filtersTitle")}
            description={t("logs.request.table.filtersDescription")}
          >
            {({ get, set }) => <RequestLogsFilters get={get} set={set} teams={teams} logsWindow={logsWindow} />}
          </DataTableFilterDrawer>
        </>
      )}
    />
  );
}
