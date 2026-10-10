"use client";

import { OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { DeletedTeam } from "@/app/(dashboard)/hooks/teams/useTeams";

import { getDeletedTeamsTableColumns } from "./DeletedTeamsTableColumns";
import { useUserDisplayNames } from "@/app/(dashboard)/hooks/users/useUsers";

interface DeletedTeamsTableProps {
  teams: DeletedTeam[];
  isLoading: boolean;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  rowCount: number;
}

const DEFAULT_SORTING: SortingState = [{ id: "deleted_at", desc: true }];

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No deleted teams found</div>
      <div className="text-sm text-muted-foreground">Teams deleted from this proxy will show up here.</div>
    </div>
  );
}

export function DeletedTeamsTable({
  teams,
  isLoading,
  pagination,
  onPaginationChange,
  rowCount,
}: DeletedTeamsTableProps) {
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const { data: displayNames } = useUserDisplayNames(
    useMemo(() => teams.map((team) => team.deleted_by ?? ""), [teams]),
  );

  const columns = useMemo(() => getDeletedTeamsTableColumns((userId) => displayNames?.[userId]), [displayNames]);

  return (
    <DataTable
      fillHeight
      data={teams}
      columns={columns}
      getRowId={(team, index) => team.team_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={rowCount}
      isLoading={isLoading}
      loadingMessage="Loading deleted teams…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
}
