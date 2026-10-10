"use client";

import { Contact } from "lucide-react";
import { useMemo } from "react";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { DataTable, DataTableViewOptions, useUrlTableState } from "@/components/shared/DataTable";
import { Input } from "@/components/ui/input";

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

export default function CustomersTable({ data, isLoading, isError, ...actions }: CustomersTableProps) {
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
          <div className="flex items-center justify-between gap-4">
            <Input
              aria-label="Search customers"
              placeholder="Search customers by ID or alias"
              className="max-w-sm"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            <DataTableViewOptions table={table} label="Columns" />
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
