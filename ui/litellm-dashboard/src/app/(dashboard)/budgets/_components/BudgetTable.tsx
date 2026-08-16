"use client";

import { Inbox } from "lucide-react";
import React, { useMemo } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";

import { getBudgetTableColumns } from "./BudgetTableColumns";

interface BudgetTableProps {
  budgets: budgetItem[];
  isLoading: boolean;
  canModify: boolean;
  onEditClick: (budget: budgetItem) => void;
  onDeleteClick: (budget: budgetItem) => void;
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No budgets yet</div>
      <div className="text-sm text-muted-foreground">
        Create a budget to set spend, TPM and RPM limits for customers.
      </div>
    </div>
  );
}

const BudgetTable: React.FC<BudgetTableProps> = ({ budgets, isLoading, canModify, onEditClick, onDeleteClick }) => {
  const rows = useMemo(
    () => [...budgets].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()),
    [budgets],
  );

  const columns = useMemo(
    () => getBudgetTableColumns({ canModify, onEditClick, onDeleteClick }),
    [canModify, onEditClick, onDeleteClick],
  );

  return (
    <DataTable
      data={rows}
      columns={columns}
      getRowId={(budget, index) => budget.budget_id || String(index)}
      isLoading={isLoading}
      loadingMessage="Loading budgets…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default BudgetTable;
