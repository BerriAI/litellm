"use client";

import { OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import { useMemo } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { DeletedKeyResponse } from "@/app/(dashboard)/hooks/keys/useKeys";

import { getDeletedKeysTableColumns } from "./DeletedKeysTableColumns";

interface DeletedKeysTableProps {
  keys: DeletedKeyResponse[];
  totalCount: number;
  isLoading: boolean;
  isError?: boolean;
  sorting: SortingState;
  onSortingChange: OnChangeFn<SortingState>;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No deleted keys found</div>
      <div className="text-sm text-muted-foreground">Keys deleted from this proxy will show up here.</div>
    </div>
  );
}

export function DeletedKeysTable({
  keys,
  totalCount,
  isLoading,
  isError,
  sorting,
  onSortingChange,
  pagination,
  onPaginationChange,
}: DeletedKeysTableProps) {
  const columns = useMemo(() => getDeletedKeysTableColumns(), []);

  return (
    <DataTable
      data={keys}
      columns={columns}
      getRowId={(key, index) => key.token || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={onSortingChange}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={totalCount}
      isLoading={isLoading}
      isError={isError}
      loadingMessage="Loading deleted keys…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
}
