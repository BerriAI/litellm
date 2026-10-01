"use client";

import { Inbox } from "lucide-react";
import React, { useMemo } from "react";

import { DataTable, useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { Guardrail } from "@/components/guardrails/types";

import { getGuardrailTableColumns } from "./guardrailTableColumns";

interface GuardrailTableProps {
  guardrailsList: Guardrail[];
  isLoading: boolean;
  onDeleteClick: (guardrailId: string, guardrailName: string) => void;
  onGuardrailClick: (id: string) => void;
}

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: ["guardrail_id", "guardrail_name", "created_at", "updated_at"],
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
      <div className="text-sm font-medium text-foreground">No guardrails yet</div>
      <div className="text-sm text-muted-foreground">Add a guardrail to start filtering requests and responses.</div>
    </div>
  );
}

const GuardrailTable: React.FC<GuardrailTableProps> = ({
  guardrailsList,
  isLoading,
  onDeleteClick,
  onGuardrailClick,
}) => {
  const { sorting, onSortingChange, pagination, onPaginationChange } = useUrlTableState(TABLE_STATE_OPTIONS);

  const columns = useMemo(
    () => getGuardrailTableColumns({ onGuardrailClick, onDeleteClick }),
    [onGuardrailClick, onDeleteClick],
  );

  return (
    <DataTable
      data={guardrailsList}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      columns={columns}
      getRowId={(guardrail, index) => guardrail.guardrail_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={onSortingChange}
      isLoading={isLoading}
      loadingMessage="Loading guardrails…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default GuardrailTable;
