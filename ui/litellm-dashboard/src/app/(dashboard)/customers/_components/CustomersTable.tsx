"use client";

import type { RowSelectionState } from "@tanstack/react-table";
import { Ban, Contact, Download, Search, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import {
  createSelectionColumn,
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableToolbar,
  useUrlTableState,
} from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { toast } from "@/lib/toast";

import { exportCustomersToCsv } from "./customerCsvExport";
import {
  CUSTOMER_FILTER_FIELDS,
  customerFilterFn,
  customerMatchesSearch,
  formatCustomerFilterValue,
} from "./customerFilters";
import { getCustomerTableColumns, type CustomerTableActions } from "./CustomersTableColumns";

const TABLE_STATE_OPTIONS = {
  sortFields: ["user_id", "spend", "max_budget", "blocked"],
  defaultSort: { id: "user_id", desc: false },
  defaultPageSize: 50,
  filterColumns: CUSTOMER_FILTER_FIELDS.map((field) => field.id),
};

const DEFAULT_COLUMN_VISIBILITY = { mcp_access: false };
const FILTER_LABELS = Object.fromEntries(CUSTOMER_FILTER_FIELDS.map((field) => [field.id, field.label]));

type CustomersTableProps = CustomerTableActions & {
  data: EndUser[];
  isLoading: boolean;
  isError: boolean;
  onRefresh: () => void;
  isRefreshing: boolean;
  onBulkSetBlocked: (userIds: string[], blocked: boolean) => Promise<boolean>;
};

function EmptyCustomers() {
  return (
    <div className="flex flex-col items-center gap-2 py-6">
      <Contact className="size-5 text-muted-foreground" />
      <p className="text-sm font-medium">No customers found</p>
      <p className="text-sm text-muted-foreground">Create a customer to manage end-user limits and access</p>
    </div>
  );
}

function BulkBlockedStateButton({
  blocked,
  premiumUser,
  disabled,
  onClick,
}: {
  blocked: boolean;
  premiumUser: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  const button = (
    <Button variant="outline" size="sm" disabled={disabled || !premiumUser} onClick={onClick}>
      {blocked ? <Ban /> : <ShieldCheck />}
      {blocked ? "Block selected" : "Unblock selected"}
    </Button>
  );
  if (premiumUser) return button;
  return (
    <TooltipProvider delay={300}>
      <Tooltip>
        <TooltipTrigger render={<span aria-label={`Bulk ${blocked ? "Block" : "Unblock"} requires Enterprise`} />}>
          {button}
        </TooltipTrigger>
        <TooltipContent role="tooltip">
          This is a LiteLLM Enterprise feature, and requires a valid key to use.
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export default function CustomersTable({
  data,
  isLoading,
  isError,
  onRefresh,
  isRefreshing,
  isBlocking,
  onBulkSetBlocked,
  ...actions
}: CustomersTableProps) {
  const {
    search,
    setSearch,
    sorting,
    onSortingChange,
    pagination,
    onPaginationChange,
    columnFilters,
    onColumnFiltersChange,
  } = useUrlTableState(TABLE_STATE_OPTIONS);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const filterScope = JSON.stringify([search, columnFilters]);
  const [selectionScope, setSelectionScope] = useState(filterScope);
  if (selectionScope !== filterScope) {
    setSelectionScope(filterScope);
    setRowSelection({});
  }
  const { canModify, premiumUser, onView, onEdit, onDelete, onSetBlocked } = actions;
  const columns = useMemo(() => {
    const columnActions = { canModify, premiumUser, isBlocking, onView, onEdit, onDelete, onSetBlocked };
    const customerColumns = getCustomerTableColumns(columnActions).map((column) =>
      CUSTOMER_FILTER_FIELDS.some((field) => field.id === column.id)
        ? { ...column, filterFn: customerFilterFn }
        : column,
    );
    return canModify
      ? [
          createSelectionColumn<EndUser>({ rowAriaLabel: (row) => `Select customer ${row.original.user_id}` }),
          ...customerColumns,
        ]
      : customerColumns;
  }, [canModify, premiumUser, isBlocking, onView, onEdit, onDelete, onSetBlocked]);
  const searchedCustomers = useMemo(
    () => data.filter((customer) => customerMatchesSearch(customer, search)),
    [data, search],
  );

  const handleBulkSetBlocked = async (userIds: string[], blocked: boolean) => {
    const canSetBlocked = canModify && !isBlocking && premiumUser;
    if (!userIds.length || !canSetBlocked) return;
    if (await onBulkSetBlocked(userIds, blocked)) setRowSelection({});
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <DataTable
        fillHeight
        pageSizeOptions={[25, 50, 100]}
        data={searchedCustomers}
        columns={columns}
        getRowId={(customer) => customer.user_id}
        defaultColumnVisibility={DEFAULT_COLUMN_VISIBILITY}
        enableRowSelection={canModify && !isBlocking}
        rowSelection={rowSelection}
        onRowSelectionChange={(updater) => {
          if (!isBlocking) setRowSelection(updater);
        }}
        toolbar={(table) => {
          const selectedIds = table.getFilteredSelectedRowModel().rows.map((row) => row.original.user_id);
          return (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <div className="relative">
                  <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    aria-label="Search customers"
                    placeholder="Search customers by ID or alias"
                    className="h-8 w-56 pl-8"
                    value={search}
                    onChange={(event) => {
                      setRowSelection({});
                      setSearch(event.target.value);
                    }}
                  />
                </div>
                <DataTableToolbar
                  className="flex-1"
                  table={table}
                  onRefresh={onRefresh}
                  isRefreshing={isRefreshing}
                  onOpenFilters={() => setFiltersOpen(true)}
                  filterLabels={FILTER_LABELS}
                  formatFilterValue={formatCustomerFilterValue}
                >
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={isLoading || isError || table.getFilteredRowModel().rows.length === 0}
                    onClick={() => {
                      try {
                        const count = exportCustomersToCsv(
                          table.getPrePaginationRowModel().rows.map((row) => row.original),
                        );
                        toast.success(`Exported ${count} customers`);
                      } catch (error) {
                        toast.fromError(error);
                      }
                    }}
                  >
                    <Download />
                    Export CSV
                  </Button>
                </DataTableToolbar>
              </div>
              {canModify && selectedIds.length > 0 && (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm text-muted-foreground">{selectedIds.length} selected</span>
                  <BulkBlockedStateButton
                    blocked
                    premiumUser={premiumUser}
                    disabled={isBlocking}
                    onClick={() => void handleBulkSetBlocked(selectedIds, true)}
                  />
                  <BulkBlockedStateButton
                    blocked={false}
                    premiumUser={premiumUser}
                    disabled={isBlocking}
                    onClick={() => void handleBulkSetBlocked(selectedIds, false)}
                  />
                  <Button variant="ghost" size="sm" disabled={isBlocking} onClick={() => setRowSelection({})}>
                    Clear selection
                  </Button>
                </div>
              )}
              <DataTableFilterDrawer
                table={table}
                open={filtersOpen}
                onOpenChange={setFiltersOpen}
                description="Filter by status, a budget limit and configured MCP grants"
              >
                {({ get, set }) =>
                  CUSTOMER_FILTER_FIELDS.map((field) => (
                    <DataTableFilterField key={field.id} label={field.label}>
                      <Select
                        value={(get(field.id) as string) || "all"}
                        onValueChange={(value) => set(field.id, value === "all" ? undefined : value)}
                      >
                        <SelectTrigger aria-label={field.label} className="w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All</SelectItem>
                          {field.options.map((option) => (
                            <SelectItem key={option.value} value={option.value}>
                              {option.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </DataTableFilterField>
                  ))
                }
              </DataTableFilterDrawer>
            </>
          );
        }}
        filterMode="client"
        columnFilters={columnFilters}
        onColumnFiltersChange={(updater) => {
          setRowSelection({});
          onColumnFiltersChange(updater);
        }}
        paginationMode="client"
        pagination={pagination}
        onPaginationChange={onPaginationChange}
        sortingMode="client"
        sorting={sorting}
        onSortingChange={onSortingChange}
        isLoading={isLoading}
        isError={isError}
        loadingMessage="Loading customers"
        noDataMessage={<EmptyCustomers />}
        size="compact"
      />
    </div>
  );
}
