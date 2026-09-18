"use client";

import { SortingState } from "@tanstack/react-table";
import { Layers } from "lucide-react";
import { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";

import { getAccessGroupsTableColumns } from "./AccessGroupsTableColumns";
import { AccessGroup } from "./types";

interface AccessGroupsTableProps {
  groups: AccessGroup[];
  isLoading: boolean;
  isFiltered: boolean;
  canModify: boolean;
  onGroupClick: (id: string) => void;
  onDeleteClick: (group: AccessGroup) => void;
}

const PAGE_SIZE_OPTIONS = [10, 25, 50];

function EmptyState({ isFiltered }: { isFiltered: boolean }) {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Layers className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {isFiltered ? "No matching access groups" : "No access groups yet"}
      </div>
      <div className="text-sm text-muted-foreground">
        {isFiltered
          ? "Try a different search term."
          : "Create an access group to manage resource permissions for your organization."}
      </div>
    </div>
  );
}

export function AccessGroupsTable({
  groups,
  isLoading,
  isFiltered,
  canModify,
  onGroupClick,
  onDeleteClick,
}: AccessGroupsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo(() => {
    const deps = { canModify, onGroupClick, onDeleteClick };
    return getAccessGroupsTableColumns(deps);
  }, [canModify, onGroupClick, onDeleteClick]);

  return (
    <DataTable
      data={groups}
      columns={columns}
      getRowId={(group, index) => group.id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      paginationMode="client"
      pageSizeOptions={PAGE_SIZE_OPTIONS}
      isLoading={isLoading}
      loadingMessage="Loading access groups…"
      noDataMessage={<EmptyState isFiltered={isFiltered} />}
      size="compact"
    />
  );
}
