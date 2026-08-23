"use client";

import { OnChangeFn, PaginationState, RowSelectionState, SortingState } from "@tanstack/react-table";
import { HeartPulse } from "lucide-react";
import { useMemo, useState } from "react";

import { Team } from "@/components/key_team_helpers/key_list";
import { DataTable } from "@/components/shared/DataTable";

import { getHealthChecksTableColumns, type HealthCheckData, type HealthStatus } from "./HealthChecksTableColumns";

interface HealthChecksTableProps {
  data: HealthCheckData[];
  rowCount: number;
  isLoading: boolean;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  rowSelection: RowSelectionState;
  onRowSelectionChange: OnChangeFn<RowSelectionState>;
  modelHealthStatuses: Record<string, HealthStatus>;
  getDisplayModelName: (model: HealthCheckData) => string;
  onRunHealthCheck: (modelId: string) => void;
  onShowError: (modelName: string, cleanedError: string, fullError: string) => void;
  onShowSuccess: (modelName: string, response: unknown) => void;
  onSelectModel?: (modelId: string) => void;
  teams?: Team[] | null;
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <HeartPulse className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No models found</div>
      <div className="text-sm text-muted-foreground">Models added to this proxy will show their health here.</div>
    </div>
  );
}

export function HealthChecksTable({
  data,
  rowCount,
  isLoading,
  pagination,
  onPaginationChange,
  rowSelection,
  onRowSelectionChange,
  modelHealthStatuses,
  getDisplayModelName,
  onRunHealthCheck,
  onShowError,
  onShowSuccess,
  onSelectModel,
  teams,
}: HealthChecksTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo(() => {
    const columnDeps = {
      modelHealthStatuses,
      getDisplayModelName,
      onRunHealthCheck,
      onShowError,
      onShowSuccess,
      onSelectModel,
      teams,
    };
    return getHealthChecksTableColumns(columnDeps);
  }, [modelHealthStatuses, getDisplayModelName, onRunHealthCheck, onShowError, onShowSuccess, onSelectModel, teams]);

  return (
    <DataTable
      data={data}
      columns={columns}
      getRowId={(row, index) => row.model_info?.id ?? String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={rowCount}
      rowSelection={rowSelection}
      onRowSelectionChange={onRowSelectionChange}
      isLoading={isLoading}
      loadingMessage="Loading models…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
}
