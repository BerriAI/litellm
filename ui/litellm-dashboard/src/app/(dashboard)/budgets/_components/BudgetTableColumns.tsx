"use client";

import { ColumnDef } from "@tanstack/react-table";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { IdCell, MoneyCell } from "@/components/shared/table_cells";
import { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

function RateLimitCell({ value }: { value: number | null }) {
  if (value == null) {
    return <span className="text-muted-foreground">n/a</span>;
  }
  return <span className="tabular-nums">{value}</span>;
}

interface BudgetRowActionsProps {
  budget: budgetItem;
  onEditClick: (budget: budgetItem) => void;
  onDeleteClick: (budget: budgetItem) => void;
}

function BudgetRowActions({ budget, onEditClick, onDeleteClick }: BudgetRowActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open budget actions"
        data-testid={`budget-actions-${budget.budget_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem data-testid="budget-action-edit" onClick={() => onEditClick(budget)}>
          <Pencil />
          Edit budget
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid="budget-action-delete"
          onClick={() => onDeleteClick(budget)}
        >
          <Trash2 />
          Delete budget
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface BudgetTableColumnsDeps {
  canModify: boolean;
  onEditClick: (budget: budgetItem) => void;
  onDeleteClick: (budget: budgetItem) => void;
}

export const getBudgetTableColumns = ({
  canModify,
  onEditClick,
  onDeleteClick,
}: BudgetTableColumnsDeps): ColumnDef<budgetItem>[] => [
  {
    id: "budget_id",
    accessorKey: "budget_id",
    meta: { title: "Budget ID" },
    header: "Budget ID",
    size: 220,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.budget_id} variant="plain" />,
  },
  {
    id: "max_budget",
    accessorKey: "max_budget",
    meta: { title: "Max Budget", numeric: true },
    header: "Max Budget",
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <MoneyCell value={row.original.max_budget} decimals={2} showZero emptyText="Unlimited" />,
  },
  {
    id: "tpm_limit",
    accessorKey: "tpm_limit",
    meta: { title: "TPM", numeric: true },
    header: "TPM",
    size: 100,
    enableSorting: false,
    cell: ({ row }) => <RateLimitCell value={row.original.tpm_limit} />,
  },
  {
    id: "rpm_limit",
    accessorKey: "rpm_limit",
    meta: { title: "RPM", numeric: true },
    header: "RPM",
    size: 100,
    enableSorting: false,
    cell: ({ row }) => <RateLimitCell value={row.original.rpm_limit} />,
  },
  ...(canModify
    ? [
        {
          id: "actions",
          meta: { className: "text-right", headerClassName: "text-right" },
          header: () => <span className="sr-only">Actions</span>,
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
