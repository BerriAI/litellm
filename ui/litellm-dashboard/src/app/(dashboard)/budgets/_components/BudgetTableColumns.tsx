"use client";

import { ColumnDef, FilterFn } from "@tanstack/react-table";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdCell, MoneyCell } from "@/components/shared/table_cells";
import type { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { buttonVariants } from "@/components/ui/button";
import { getBudgetDurationLabel } from "@/components/common_components/budget_duration_dropdown";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import type { TFunction } from "i18next";

/**
 * Filtering happens on the server, so this never runs as a predicate. It exists to override
 * TanStack's auto-remove heuristic, which infers a filter shape from the column's first cell
 * and silently discards a filter whose value is not that shape (a range object on a numeric
 * column, for instance).
 */
const serverFilter: FilterFn<budgetItem> = () => true;
serverFilter.autoRemove = () => false;

function RateLimitCell({ value, t }: { value: number | null | undefined; t: TFunction<"gateway"> }) {
  if (value == null) {
    return <span className="text-muted-foreground">{t("budgets.table.notAvailable")}</span>;
  }
  return <span className="tabular-nums">{value}</span>;
}

function BudgetDurationCell({ value, t }: { value: string | null | undefined; t: TFunction<"gateway"> }) {
  if (!value) {
    return <span className="text-muted-foreground">{t("budgets.table.notSet")}</span>;
  }
  return <span className="whitespace-nowrap">{getBudgetDurationLabel(value, t)}</span>;
}

interface BudgetRowActionsProps {
  budget: budgetItem;
  onEditClick: (budget: budgetItem) => void;
  onDeleteClick: (budget: budgetItem) => void;
  t: TFunction<"gateway">;
}

function BudgetRowActions({ budget, onEditClick, onDeleteClick, t }: BudgetRowActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("budgets.table.openActions")}
        data-testid={`budget-actions-${budget.budget_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem data-testid="budget-action-edit" onClick={() => onEditClick(budget)}>
          <Pencil />
          {t("budgets.table.edit")}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid="budget-action-delete"
          onClick={() => onDeleteClick(budget)}
        >
          <Trash2 />
          {t("budgets.table.delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** Off by default so the table opens on the four columns it has always shown; the Columns menu turns them on. */
export const BUDGET_TABLE_HIDDEN_COLUMNS: Record<string, boolean> = {
  budget_duration: false,
  created_at: false,
};

interface BudgetTableColumnsDeps {
  canModify: boolean;
  onEditClick: (budget: budgetItem) => void;
  onDeleteClick: (budget: budgetItem) => void;
  t: TFunction<"gateway">;
}

export const getBudgetTableColumns = ({
  canModify,
  onEditClick,
  onDeleteClick,
  t,
}: BudgetTableColumnsDeps): ColumnDef<budgetItem>[] => [
  {
    id: "budget_id",
    accessorKey: "budget_id",
    meta: { title: t("budgets.fields.budgetId") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.fields.budgetId")} />,
    cell: ({ row }) => (
      <IdCell value={row.original.budget_id} variant="plain" truncate={false} copyable className="whitespace-nowrap" />
    ),
  },
  {
    id: "max_budget",
    accessorKey: "max_budget",
    filterFn: serverFilter,
    meta: { title: t("budgets.fields.maxBudget"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.fields.maxBudget")} />,
    size: 120,
    cell: ({ row }) => (
      <MoneyCell value={row.original.max_budget} decimals={2} showZero emptyText={t("budgets.table.unlimited")} />
    ),
  },
  {
    id: "tpm_limit",
    accessorKey: "tpm_limit",
    meta: { title: "TPM", numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title="TPM" />,
    size: 100,
    cell: ({ row }) => <RateLimitCell value={row.original.tpm_limit} t={t} />,
  },
  {
    id: "rpm_limit",
    accessorKey: "rpm_limit",
    meta: { title: "RPM", numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title="RPM" />,
    size: 100,
    cell: ({ row }) => <RateLimitCell value={row.original.rpm_limit} t={t} />,
  },
  {
    id: "budget_duration",
    accessorKey: "budget_duration",
    filterFn: serverFilter,
    meta: { title: t("budgets.fields.reset") },
    // "7d"/"30d" sort lexicographically, not chronologically, so the route does not offer it.
    enableSorting: false,
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.fields.reset")} />,
    size: 110,
    cell: ({ row }) => <BudgetDurationCell value={row.original.budget_duration} t={t} />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    filterFn: serverFilter,
    meta: { title: t("budgets.fields.created") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.fields.created")} />,
    size: 160,
    cell: ({ row }) => <DateCell value={row.original.created_at} />,
  },
  ...(canModify
    ? [
        {
          id: "actions",
          meta: { className: "text-right", headerClassName: "text-right" },
          header: () => <span className="sr-only">{t("budgets.fields.actions")}</span>,
          size: 64,
          enableSorting: false,
          enableHiding: false,
          cell: ({ row }) => (
            <div className="flex justify-end">
              <BudgetRowActions budget={row.original} onEditClick={onEditClick} onDeleteClick={onDeleteClick} t={t} />
            </div>
          ),
        } satisfies ColumnDef<budgetItem>,
      ]
    : []),
];
