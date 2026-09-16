"use client";

import { Inbox } from "lucide-react";
import React, { useMemo } from "react";

import { DataTable, useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { Tag } from "@/components/tag_management/types";

import { getTagTableColumns } from "./tagTableColumns";

interface TagTableProps {
  data: Tag[];
  onEdit: (tag: Tag) => void;
  onDelete: (tagName: string) => void;
  onSelectTag: (tagName: string) => void;
  isLoading?: boolean;
}

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: ["name", "created_at"],
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
      <div className="text-sm font-medium text-foreground">No tags yet</div>
      <div className="text-sm text-muted-foreground">Create a tag to start routing and restricting model usage.</div>
    </div>
  );
}

const TagTable: React.FC<TagTableProps> = ({ data, onEdit, onDelete, onSelectTag, isLoading = false }) => {
  const { sorting, onSortingChange, pagination, onPaginationChange } = useUrlTableState(TABLE_STATE_OPTIONS);

  const columns = useMemo(() => getTagTableColumns({ onSelectTag, onEdit, onDelete }), [onSelectTag, onEdit, onDelete]);

  return (
    <DataTable
      data={data}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      columns={columns}
      getRowId={(tag, index) => tag.name || String(index)}
      fillHeight
      sortingMode="client"
      sorting={sorting}
      onSortingChange={onSortingChange}
      isLoading={isLoading}
      loadingMessage="Loading tags…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default TagTable;
