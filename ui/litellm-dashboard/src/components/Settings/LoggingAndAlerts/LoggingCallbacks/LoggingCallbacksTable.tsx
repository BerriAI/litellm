"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { Inbox, Plus, SearchX } from "lucide-react";
import React, { useMemo } from "react";

import {
  DataTable,
  DataTableSortHeader,
  DataTableToolbar,
  useUrlTableState,
  type UrlTableStateOptions,
} from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";

import {
  AvailableCallbacks,
  CallbackRow,
  callbackRowMode,
  getLoggingCallbacksTableColumns,
} from "./LoggingCallbacksTableColumns";
import { AlertingObject } from "./types";

type LoggingCallbacksProps = {
  callbacks: AlertingObject[];
  availableCallbacks?: AvailableCallbacks;
  isLoading?: boolean;
  onTest?: (callback: AlertingObject) => void | Promise<void>;
  onEdit?: (callback: AlertingObject) => void;
  onDelete?: (callback: AlertingObject) => void;
  onAdd?: () => void;
};

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: ["name", "mode"],
  defaultSort: { id: "name", desc: false },
  defaultPageSize: 25,
  filterColumns: [],
  keyPrefix: "callbacks_",
};

const callbackDisplayName = (callback: CallbackRow, availableCallbacks: AvailableCallbacks): string =>
  availableCallbacks[callback.name]?.ui_callback_name || callback.name;

const withSortableColumns = (
  columns: ColumnDef<CallbackRow>[],
  availableCallbacks: AvailableCallbacks,
): ColumnDef<CallbackRow>[] => {
  const sortValues: Record<string, (callback: CallbackRow) => string> = {
    name: (callback) => callbackDisplayName(callback, availableCallbacks),
    mode: callbackRowMode,
  };
  return columns.map((column): ColumnDef<CallbackRow> => {
    const id = column.id;
    if (id === undefined) return column;
    const accessorFn = sortValues[id];
    if (accessorFn === undefined) return column;
    const title = column.meta?.title ?? id;
    return {
      ...column,
      id,
      accessorFn,
      enableSorting: true,
      sortingFn: "alphanumeric",
      header: ({ column: sortable }) => <DataTableSortHeader column={sortable} title={title} />,
    };
  });
};

function NoMatchesState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <SearchX className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No matching callbacks</div>
      <div className="text-sm text-muted-foreground">Try a different search term.</div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No callbacks configured</div>
      <div className="text-sm text-muted-foreground">
        Add your first callback to start logging data to external services.
      </div>
    </div>
  );
}

export const LoggingCallbacksTable: React.FC<LoggingCallbacksProps> = ({
  callbacks,
  availableCallbacks = {},
  isLoading = false,
  onTest = () => {},
  onEdit = () => {},
  onDelete = () => {},
  onAdd = () => {},
}) => {
  const { search, setSearch, sorting, onSortingChange, pagination, onPaginationChange } =
    useUrlTableState(TABLE_STATE_OPTIONS);

  const columns = useMemo(() => {
    const deps = { availableCallbacks, onTest, onEdit, onDelete };
    return withSortableColumns(getLoggingCallbacksTableColumns(deps), availableCallbacks);
  }, [availableCallbacks, onTest, onEdit, onDelete]);

  const query = search.trim().toLowerCase();
  const visibleCallbacks = useMemo<CallbackRow[]>(() => {
    if (!query) return callbacks;
    return callbacks.filter((callback) =>
      [callback.name, callbackDisplayName(callback, availableCallbacks), callbackRowMode(callback)].some((value) =>
        value.toLowerCase().includes(query),
      ),
    );
  }, [callbacks, availableCallbacks, query]);

  return (
    <div className="mt-4 flex w-full flex-col gap-4">
      <h3 className="text-lg font-semibold tracking-tight text-foreground">Active Logging Callbacks</h3>
      <div>
        <Button onClick={onAdd}>
          <Plus />
          Add Callback
        </Button>
      </div>
      <DataTable
        data={visibleCallbacks}
        columns={columns}
        getRowId={(callback, index) => `${callback.name || index}-${callbackRowMode(callback)}`}
        sortingMode="client"
        sorting={sorting}
        onSortingChange={onSortingChange}
        paginationMode="client"
        pagination={pagination}
        onPaginationChange={onPaginationChange}
        toolbar={(table) => (
          <DataTableToolbar
            table={table}
            searchValue={search}
            onSearchChange={setSearch}
            searchPlaceholder="Search callbacks"
            showViewOptions={false}
          />
        )}
        isLoading={isLoading}
        loadingMessage="Loading callbacks…"
        noDataMessage={query ? <NoMatchesState /> : <EmptyState />}
        size="compact"
      />
    </div>
  );
};
