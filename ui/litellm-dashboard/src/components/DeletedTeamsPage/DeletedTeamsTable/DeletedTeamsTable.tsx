"use client";

import { OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { DataTable } from "@/components/shared/DataTable";
import { DeletedTeam } from "@/app/(dashboard)/hooks/teams/useTeams";

import { getDeletedTeamsTableColumns } from "./DeletedTeamsTableColumns";

interface DeletedTeamsTableProps {
  teams: DeletedTeam[];
  isLoading: boolean;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  rowCount: number;
}

const DEFAULT_SORTING: SortingState = [{ id: "deleted_at", desc: true }];

function EmptyState() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("logs.deletedTeams.table.emptyTitle")}</div>
      <div className="text-sm text-muted-foreground">{t("logs.deletedTeams.table.emptyDescription")}</div>
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
  const { t } = useTranslation();
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(() => getDeletedTeamsTableColumns(t), [t]);

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
      loadingMessage={t("logs.deletedTeams.table.loading")}
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
}
