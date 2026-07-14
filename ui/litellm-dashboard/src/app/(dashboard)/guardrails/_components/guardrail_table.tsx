"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useMemo, useState } from "react";

import { DataTable, DataTableToolbar } from "@/components/shared/DataTable";
import { Guardrail } from "@/components/guardrails/types";

import { getGuardrailTableColumns } from "./guardrailTableColumns";

interface GuardrailTableProps {
  guardrailsList: Guardrail[];
  isLoading: boolean;
  onDeleteClick: (guardrailId: string, guardrailName: string) => void;
  onGuardrailUpdated: () => void;
  onGuardrailClick: (id: string) => void;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

function EmptyState({ searching }: { searching: boolean }) {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {searching ? "No matching guardrails" : "No guardrails yet"}
      </div>
      <div className="text-sm text-muted-foreground">
        {searching ? "Try a different name or ID." : "Add a guardrail to start filtering requests and responses."}
      </div>
    </div>
  );
}

const GuardrailTable: React.FC<GuardrailTableProps> = ({
  guardrailsList,
  isLoading,
  onDeleteClick,
  onGuardrailUpdated,
  onGuardrailClick,
}) => {
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const [search, setSearch] = useState("");

  const columns = useMemo(
    () => getGuardrailTableColumns({ onGuardrailClick, onDeleteClick }),
    [onGuardrailClick, onDeleteClick],
  );

  const filteredGuardrails = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return guardrailsList;
    return guardrailsList.filter(
      (guardrail) =>
        (guardrail.guardrail_name ?? "").toLowerCase().includes(query) ||
        (guardrail.guardrail_id ?? "").toLowerCase().includes(query),
    );
  }, [guardrailsList, search]);

  return (
    <DataTable
      data={filteredGuardrails}
      columns={columns}
      getRowId={(guardrail, index) => guardrail.guardrail_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage="Loading guardrails…"
      noDataMessage={<EmptyState searching={search.trim().length > 0} />}
      size="compact"
      toolbar={(table) => (
        <DataTableToolbar
          table={table}
          searchValue={search}
          onSearchChange={setSearch}
          searchPlaceholder="Search guardrails by name or ID…"
          onRefresh={onGuardrailUpdated}
          showViewOptions={false}
        />
      )}
    />
  );
};

export default GuardrailTable;
