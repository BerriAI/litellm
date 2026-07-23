"use client";

import type { ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { ScrollText } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { DataTable, DataTableFilterDrawer, DataTableToolbar } from "@/components/shared/DataTable";

import type { Team } from "../key_team_helpers/key_list";
import type { LogEntry } from "./columns";
import { LOG_FILTER_LABELS } from "./log_filter_logic";
import { RequestLogsFilters } from "./RequestLogsFilters";
import { getRequestLogsTableColumns } from "./RequestLogsTableColumns";

interface RequestLogsTableProps {
  data: LogEntry[];
  rowCount: number;
  isLoading: boolean;
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
  onSessionClick: (sessionId: string) => void;
  teams: Team[];
  accessToken: string;
  toolbarChildren?: ReactNode;
}

function RequestLogsEmptyState({ filtered }: { filtered: boolean }) {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <ScrollText className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{filtered ? "No matching requests" : "No requests yet"}</div>
      <div className="max-w-xs text-center text-sm text-muted-foreground">
        {filtered
          ? "No requests match your filters for this time range."
          : "Requests proxied through LiteLLM will appear here."}
      </div>
    </div>
  );
}

export function RequestLogsTable({
  data,
  rowCount,
  isLoading,
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
  accessToken,
  toolbarChildren,
}: RequestLogsTableProps) {
  const [filtersOpen, setFiltersOpen] = useState(false);

  const columns = useMemo(() => {
    const deps = { onKeyHashClick, onSessionClick };
    return getRequestLogsTableColumns(deps);
  }, [onKeyHashClick, onSessionClick]);

  const isFiltered = columnFilters.length > 0 || searchValue !== "";

  return (
    <DataTable
      data={data}
      columns={columns}
      getRowId={(row) => row.request_id}
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
      loadingMessage="Loading request logs…"
      noDataMessage={<RequestLogsEmptyState filtered={isFiltered} />}
      size="compact"
      onRowClick={onRowClick}
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={searchValue}
            onSearchChange={onSearchChange}
            searchPlaceholder="Search by Request ID"
            onRefresh={onRefresh}
            isRefreshing={isRefreshing}
            onOpenFilters={() => setFiltersOpen(true)}
            filterLabels={LOG_FILTER_LABELS}
            showViewOptions={false}
          >
            {toolbarChildren}
          </DataTableToolbar>
          <DataTableFilterDrawer
            table={table}
            open={filtersOpen}
            onOpenChange={setFiltersOpen}
            title="Filters"
            description="Narrow down request logs"
          >
            {({ get, set }) => <RequestLogsFilters get={get} set={set} teams={teams} accessToken={accessToken} />}
          </DataTableFilterDrawer>
        </>
      )}
    />
  );
}
