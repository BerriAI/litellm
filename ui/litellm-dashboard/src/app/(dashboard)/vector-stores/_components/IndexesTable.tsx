"use client";

import { Inbox } from "lucide-react";
import React, { useMemo } from "react";

import { DataTable, useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";

import type { VectorStoreIndex } from "./IndexesTab";
import { getIndexesTableColumns } from "./IndexesTableColumns";

interface IndexesTableProps {
  data: VectorStoreIndex[];
  resolveVectorStoreId: (name: string) => string | undefined;
  onViewVectorStore: (vectorStoreId: string) => void;
  isLoading?: boolean;
}

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: ["index_name", "vector_store_name", "created_at"],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 25,
  filterColumns: [],
  keyPrefix: "idx_",
};

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No indexes registered yet</div>
      <div className="text-sm text-muted-foreground">Indexes registered on this proxy will appear here.</div>
    </div>
  );
}

const IndexesTable: React.FC<IndexesTableProps> = ({
  data,
  resolveVectorStoreId,
  onViewVectorStore,
  isLoading = false,
}) => {
  const { sorting, onSortingChange, pagination, onPaginationChange } = useUrlTableState(TABLE_STATE_OPTIONS);

  const columns = useMemo(
    () => getIndexesTableColumns({ resolveVectorStoreId, onViewVectorStore }),
    [resolveVectorStoreId, onViewVectorStore],
  );

  return (
    <DataTable
      data={data}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      columns={columns}
      getRowId={(row, index) => row.id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={onSortingChange}
      isLoading={isLoading}
      loadingMessage="Loading indexes…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default IndexesTable;
