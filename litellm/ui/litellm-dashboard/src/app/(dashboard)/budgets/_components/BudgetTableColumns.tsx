"use client";

import { ColumnDef, FilterFn } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

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

/**
 * Filtering happens on the server, so this never runs as a predicate. It exists to override
 * TanStack's auto-remove heuristic, which infers a filter shape from the column's first cell
 * and silently discards a filter whose value is not that shape (a range object on a numeric
 * column, for instance).
 */
const serverFilter: FilterFn<budgetItem> = () => true;
serverFilter.autoRemove = () => false;

function RateLimitCell({ value }: { value: number | null | undefined }) {
  const { t } = useTranslation();
  if (value == null) {
    return <span className="text-muted-foreground">{t("budgets.table.cells.notApplicable")}</span>;
  }
  return <span className="tabular-nums">{value}</span>;
}

function BudgetDurationCell({ value }: { value: string | null | undefined }) {
  const { t } = useTranslation();
  if (!value) {
    return <span className="text-muted-foreground">{t("budgets.table.cells.notSet")}</span>;
  }
  return <span className="whitespace-nowrap">{getBudgetDurationLabel(value)}</span>;
}

function BudgetLimitsCell({ budget }: { budget: budgetItem }) {
  const { t } = useTranslation();
  const periodLabel = (duration: string) => {
    if (duration === "1mo") return t("budgets.form.resetOptions.perMonth");
    if (duration === "12mo") return t("budgets.form.resetOptions.perYear");
    return getBudgetDurationLabel(duration);
  };
  return (
    <div className="space-y-1 text-right tabular-nums">
      {(budget.max_budget != null || !budget.budget_limits?.length) && (
        <MoneyCell value={budget.max_budget} decimals={2} showZero emptyText={t("budgets.table.cells.unlimited")} />
      )}
      {budget.budget_limits?.map((limit) => (
        <div key={limit.budget_duration} className="whitespace-nowrap">
          {t("budgets.table.cells.windowLimit", {
            amount: limit.max_budget,
            period: periodLabel(limit.budget_duration),
          })}
        </div>
      ))}
    </div>
  );
}

interface BudgetRowActionsProps {
  budget: budgetItem;
  onEditClick: (budget: budgetItem) => void;
  onDeleteClick: (budget: budgetItem) => void;
}

function BudgetRowActions({ budget, onEditClick, onDeleteClick }: BudgetRowActionsProps) {
  const { t } = useTranslation();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("budgets.table.rowActions.open")}
        data-testid={`budget-actions-${budget.budget_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem data-testid="budget-action-edit" onClick={() => onEditClick(budget)}>
          <Pencil />
          {t("budgets.table.rowActions.edit")}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid="budget-action-delete"
          onClick={() => onDeleteClick(budget)}
        >
          <Trash2 />
          {t("budgets.table.rowActions.delete")}
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
}

export const getBudgetTableColumns = (
  { canModify, onEditClick, onDeleteClick }: BudgetTableColumnsDeps,
  t: TFunction,
): ColumnDef<budgetItem>[] => [
  {
    id: "budget_id",
    accessorKey: "budget_id",
    meta: { title: t("budgets.table.columns.budgetId") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.table.columns.budgetId")} />,
    cell: ({ row }) => (
      <IdCell value={row.original.budget_id} variant="plain" truncate={false} copyable className="whitespace-nowrap" />
    ),
  },
  {
    id: "max_budget",
    accessorKey: "max_budget",
    filterFn: serverFilter,
    meta: { title: t("budgets.table.columns.maxBudget"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.table.columns.maxBudget")} />,
    size: 120,
    cell: ({ row }) => <BudgetLimitsCell budget={row.original} />,
  },
  {
    id: "tpm_limit",
    accessorKey: "tpm_limit",
    meta: { title: t("budgets.table.columns.tpm"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.table.columns.tpm")} />,
    size: 100,
    cell: ({ row }) => <RateLimitCell value={row.original.tpm_limit} />,
  },
  {
    id: "rpm_limit",
    accessorKey: "rpm_limit",
    meta: { title: t("budgets.table.columns.rpm"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.table.columns.rpm")} />,
    size: 100,
    cell: ({ row }) => <RateLimitCell value={row.original.rpm_limit} />,
  },
  {
    id: "tpd_limit",
    accessorKey: "tpd_limit",
    meta: { title: t("budgets.table.columns.tpd"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.table.columns.tpd")} />,
    size: 110,
    cell: ({ row }) => <RateLimitCell value={row.original.tpd_limit} />,
  },
  {
    id: "budget_duration",
    accessorKey: "budget_duration",
    filterFn: serverFilter,
    meta: { title: t("budgets.table.columns.reset") },
    sortDescFirst: false,
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.table.columns.reset")} />,
    size: 110,
    cell: ({ row }) => <BudgetDurationCell value={row.original.budget_duration} />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    filterFn: serverFilter,
    meta: { title: t("budgets.table.columns.created") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("budgets.table.columns.created")} />,
    size: 160,
    cell: ({ row }) => <DateCell value={row.original.created_at} />,
  },
  ...(canModify
    ? [
        {
          id: "actions",
          meta: { className: "text-right", headerClassName: "text-right" },
          header: () => <span className="sr-only">{t("budgets.table.columns.actions")}</span>,
          size: 64,
          enableSorting: false,
          enableHiding: false,
          cell: ({ row }) => (
            <div className="flex justify-end">
              <BudgetRowActions budget={row.original} onEditClick={onEditClick} onDeleteClick={onDeleteClick} />
            </div>
          ),
        } satisfies ColumnDef<budgetItem>,
      ]
    : []),
];
