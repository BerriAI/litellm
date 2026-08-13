"use client";

import { OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import { useMemo, useState } from "react";

import type { CLISessionResponse } from "@/app/(dashboard)/hooks/cliSessions/useCLISessions";
import { DataTable } from "@/components/shared/DataTable";

import { getCLISessionsTableColumns } from "./CLISessionsTableColumns";

interface CLISessionsTableProps {
  sessions: CLISessionResponse[];
  totalCount: number;
  isLoading: boolean;
  isRevoking: boolean;
  canRevoke: boolean;
  onRevoke: (sessionId: string) => void;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No active CLI sessions</div>
      <div className="text-sm text-muted-foreground">Sessions created by `lite login` will show up here.</div>
    </div>
  );
}

export function CLISessionsTable({
  sessions,
  totalCount,
  isLoading,
  isRevoking,
  canRevoke,
  onRevoke,
  pagination,
  onPaginationChange,
}: CLISessionsTableProps) {
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(
    () => getCLISessionsTableColumns(onRevoke, isRevoking, canRevoke),
    [onRevoke, isRevoking, canRevoke],
  );

  return (
    <DataTable
      data={sessions}
      columns={columns}
      getRowId={(session, index) => session.session_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={totalCount}
      isLoading={isLoading}
      loadingMessage="Loading CLI sessions…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
}
