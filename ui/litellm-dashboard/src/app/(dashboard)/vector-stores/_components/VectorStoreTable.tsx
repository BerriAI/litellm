"use client";

import { Inbox } from "lucide-react";
import React, { useMemo } from "react";

import { DataTable, useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { VectorStore } from "@/components/vector_store_management/types";

import { getVectorStoreTableColumns } from "./VectorStoreTableColumns";

interface VectorStoreTableProps {
  data: VectorStore[];
  onView: (vectorStoreId: string) => void;
  onEdit: (vectorStoreId: string) => void;
  onDelete: (vectorStoreId: string) => void;
  isLoading?: boolean;
}

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: ["vector_store_id", "vector_store_name", "created_at", "updated_at"],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 25,
  filterColumns: [],
};

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No vector stores</div>
      <div className="text-sm text-muted-foreground">
        Connect a vector store to enable retrieval-augmented generation.
      </div>
    </div>
  );
}

const VectorStoreTable: React.FC<VectorStoreTableProps> = ({ data, onView, onEdit, onDelete, isLoading = false }) => {
  const { sorting, onSortingChange, pagination, onPaginationChange } = useUrlTableState(TABLE_STATE_OPTIONS);

  const columns = useMemo(() => getVectorStoreTableColumns({ onView, onEdit, onDelete }), [onView, onEdit, onDelete]);

  return (
    <DataTable
      data={data}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      columns={columns}
      getRowId={(vectorStore, index) => vectorStore.vector_store_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={onSortingChange}
      isLoading={isLoading}
      loadingMessage="Loading vector stores…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default VectorStoreTable;
