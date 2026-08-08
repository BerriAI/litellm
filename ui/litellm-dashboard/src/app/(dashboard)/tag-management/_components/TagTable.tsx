"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { Tag } from "@/components/tag_management/types";

import { getTagTableColumns } from "./tagTableColumns";

interface TagTableProps {
  data: Tag[];
  onEdit: (tag: Tag) => void;
  onDelete: (tagName: string) => void;
  onSelectTag: (tagName: string) => void;
  isLoading?: boolean;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

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
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(() => getTagTableColumns({ onSelectTag, onEdit, onDelete }), [onSelectTag, onEdit, onDelete]);

  return (
    <DataTable
      data={data}
      columns={columns}
      getRowId={(tag, index) => tag.name || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage="Loading tags…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default TagTable;
