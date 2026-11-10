"use client";

import { SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { DataTable } from "@/components/shared/DataTable";
import { Guardrail } from "@/components/guardrails/types";

import { getGuardrailTableColumns } from "./guardrailTableColumns";

interface GuardrailTableProps {
  guardrailsList: Guardrail[];
  isLoading: boolean;
  onDeleteClick: (guardrailId: string, guardrailName: string) => void;
  onGuardrailClick: (id: string) => void;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

function EmptyState() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("guardrails.table.empty.title")}</div>
      <div className="text-sm text-muted-foreground">{t("guardrails.table.empty.description")}</div>
    </div>
  );
}

const GuardrailTable: React.FC<GuardrailTableProps> = ({
  guardrailsList,
  isLoading,
  onDeleteClick,
  onGuardrailClick,
}) => {
  const { t } = useTranslation();
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(
    () => getGuardrailTableColumns({ onGuardrailClick, onDeleteClick, t }),
    [onGuardrailClick, onDeleteClick, t],
  );

  return (
    <DataTable
      data={guardrailsList}
      paginationMode="client"
      columns={columns}
      getRowId={(guardrail, index) => guardrail.guardrail_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage={t("guardrails.table.loading")}
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default GuardrailTable;
