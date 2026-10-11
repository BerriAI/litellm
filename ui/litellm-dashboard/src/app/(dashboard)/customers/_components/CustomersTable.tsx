"use client";

import { Contact, RefreshCw, Search } from "lucide-react";
import { useMemo } from "react";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { DataTable, DataTableViewOptions, useUrlTableState } from "@/components/shared/DataTable";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

import { getCustomerTableColumns, type CustomerTableActions } from "./CustomersTableColumns";

const TABLE_STATE_OPTIONS = {
  sortFields: ["user_id", "spend", "max_budget", "blocked"],
  defaultSort: { id: "user_id", desc: false },
  defaultPageSize: 50,
  filterColumns: [],
};

const DEFAULT_COLUMN_VISIBILITY = { mcp_access: false };

type CustomersTableProps = CustomerTableActions & {
  data: EndUser[];
  isLoading: boolean;
  isError: boolean;
  onRefresh: () => void;
  isRefreshing: boolean;
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

export default function CustomersTable({
  data,
  isLoading,
  isError,
  onRefresh,
  isRefreshing,
  ...actions
}: CustomersTableProps) {
  const { search, setSearch, sorting, onSortingChange, pagination, onPaginationChange } =
    useUrlTableState(TABLE_STATE_OPTIONS);
  const { canModify, premiumUser, onView, onEdit, onDelete, onSetBlocked } = actions;
  const columns = useMemo(() => {
    const columnActions = { canModify, premiumUser, onView, onEdit, onDelete, onSetBlocked };
    return getCustomerTableColumns(columnActions);
  }, [canModify, premiumUser, onView, onEdit, onDelete, onSetBlocked]);
  const filteredCustomers = useMemo(() => {
    const query = search.trim().toLowerCase();
    return data.filter((customer) => `${customer.user_id} ${customer.alias ?? ""}`.toLowerCase().includes(query));
  }, [data, search]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <DataTable
        fillHeight
        pageSizeOptions={[25, 50, 100]}
        data={filteredCustomers}
        columns={columns}
        getRowId={(customer) => customer.user_id}
        defaultColumnVisibility={DEFAULT_COLUMN_VISIBILITY}
        toolbar={(table) => (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="relative">
              <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                aria-label="Search customers"
                placeholder="Search customers by ID or alias"
                className="h-8 w-56 pl-8"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="outline"
                size="icon-sm"
                onClick={onRefresh}
                disabled={isRefreshing}
                aria-label="Refresh"
                title="Refresh"
              >
                <RefreshCw className={isRefreshing ? "animate-spin" : ""} />
              </Button>
              <DataTableViewOptions table={table} label="Columns" />
            </div>
          </div>
        )}
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
