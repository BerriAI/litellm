"use client";

import { useMemo } from "react";

import { useKeyBudgets, type KeyBudgetEntry } from "@/app/(dashboard)/hooks/keys/useKeyBudgets";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { DataTable } from "@/components/shared/DataTable";
import { cn } from "@/lib/cva.config";

import { parseErrorMessage } from "../shared/errorUtils";
import { KeyBudgetsBulletChart } from "./KeyBudgetsBulletChart";
import { getKeyBudgetsTableColumns, isBlockingRow, rowRank } from "./KeyBudgetsTableColumns";

function BudgetRows({ budgets, isLoading }: { budgets: readonly KeyBudgetEntry[]; isLoading: boolean }) {
  const columns = useMemo(() => getKeyBudgetsTableColumns(), []);
  const rows = useMemo(() => [...budgets].sort((a, b) => rowRank(a) - rowRank(b)), [budgets]);

  return (
    <>
      {!isLoading && budgets.length > 0 && <KeyBudgetsBulletChart budgets={budgets} />}
      <DataTable
        data={rows}
        columns={columns}
        getRowId={(entry, index) => `${entry.scope}:${entry.entity_id ?? ""}:${index}`}
        isLoading={isLoading}
        loadingMessage="Loading budgets…"
        noDataMessage="No budgets apply to this key."
        // Every row is two lines tall whatever it carries, so the table reads as a grid rather than
        // taking its height from whichever caveat happened to be longest.
        rowClassName={(row) => cn("h-14", isBlockingRow(row.original) ? "bg-red-50 hover:bg-red-50" : "")}
        size="compact"
      />
    </>
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
