"use client";

import { useMemo } from "react";

import { useKeyBudgets, type KeyBudgetEntry } from "@/app/(dashboard)/hooks/keys/useKeyBudgets";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { DataTable } from "@/components/shared/DataTable";

import { parseErrorMessage } from "../shared/errorUtils";
import { getKeyBudgetsTableColumns, isBlockingRow, severityRank } from "./KeyBudgetsTableColumns";

function BudgetRows({ budgets, isLoading }: { budgets: readonly KeyBudgetEntry[]; isLoading: boolean }) {
  const columns = useMemo(() => getKeyBudgetsTableColumns(), []);
  const rows = useMemo(() => [...budgets].sort((a, b) => severityRank(a) - severityRank(b)), [budgets]);

  return (
    <DataTable
      data={rows}
      columns={columns}
      getRowId={(entry, index) => `${entry.scope}:${entry.entity_id ?? ""}:${index}`}
      isLoading={isLoading}
      loadingMessage="Loading budgets…"
      noDataMessage="No budgets apply to this key."
      rowClassName={(row) => (isBlockingRow(row.original) ? "bg-red-50 hover:bg-red-50" : "")}
      size="compact"
    />
  );
}

export function KeyBudgetsTable({ keyId }: { keyId: string }) {
  const { data, isLoading, isError, error } = useKeyBudgets(keyId);

  return (
    <div data-testid="key-budgets-panel" className="flex flex-col gap-3">
      <p className="text-sm text-muted-foreground">
        Every budget that can block this key, with the live spend each one is measured against.
      </p>
      {isError ? (
        <Alert variant="error">
          <AlertTitle>Could not load budgets</AlertTitle>
          <AlertDescription>{parseErrorMessage(error)}</AlertDescription>
        </Alert>
      ) : (
        <BudgetRows budgets={data?.budgets ?? []} isLoading={isLoading} />
      )}
    </div>
  );
}
