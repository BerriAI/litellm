"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { DeletedTeam } from "@/app/(dashboard)/hooks/teams/useTeams";

import { getDeletedTeamsTableColumns } from "./DeletedTeamsTableColumns";

interface DeletedTeamsTableProps {
  teams: DeletedTeam[];
  isLoading: boolean;
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

export function DeletedTeamsTable({ teams, isLoading }: DeletedTeamsTableProps) {
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(() => getDeletedTeamsTableColumns(), []);

  return (
    <DataTable
      data={teams}
      columns={columns}
      getRowId={(team, index) => team.team_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage="Loading deleted teams…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
}
