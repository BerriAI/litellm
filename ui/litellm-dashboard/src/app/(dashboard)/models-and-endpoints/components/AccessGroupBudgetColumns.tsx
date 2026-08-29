"use client";

import { ColumnDef } from "@tanstack/react-table";
import { MoreHorizontal, Trash2, Wallet } from "lucide-react";

import { getBudgetDurationLabel } from "@/components/common_components/budget_duration_dropdown";
import { DataTableSortHeader } from "@/components/shared/DataTable";
import { ModelsCell, SpendBudgetCell } from "@/components/shared/table_cells";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import { ModelAccessGroup } from "@/app/(dashboard)/hooks/modelAccessGroups/useModelAccessGroups";

interface AccessGroupRowActionsProps {
  accessGroup: ModelAccessGroup;
  onSetBudget: (accessGroup: ModelAccessGroup) => void;
  onClearBudget: (accessGroup: ModelAccessGroup) => void;
}

function AccessGroupRowActions({ accessGroup, onSetBudget, onClearBudget }: AccessGroupRowActionsProps) {
  const hasBudget = accessGroup.budget != null;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={`Open budget actions for ${accessGroup.access_group}`}
        data-testid={`access-group-actions-${accessGroup.access_group}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem data-testid="access-group-action-set-budget" onClick={() => onSetBudget(accessGroup)}>
          <Wallet />
          {hasBudget ? "Edit budget" : "Set budget"}
        </DropdownMenuItem>
        <DropdownMenuItem
          variant="destructive"
          disabled={!hasBudget}
          data-testid="access-group-action-clear-budget"
          title={hasBudget ? undefined : "This access group has no budget to clear"}
          onClick={() => onClearBudget(accessGroup)}
        >
          <Trash2 />
          Clear budget
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface AccessGroupBudgetColumnsDeps {
  onSetBudget: (accessGroup: ModelAccessGroup) => void;
  onClearBudget: (accessGroup: ModelAccessGroup) => void;
}

export const getAccessGroupBudgetColumns = ({
  onSetBudget,
  onClearBudget,
}: AccessGroupBudgetColumnsDeps): ColumnDef<ModelAccessGroup>[] => [
  {
    id: "access_group",
    accessorKey: "access_group",
    meta: { title: "Access Group" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Access Group" />,
    size: 220,
    enableSorting: true,
    cell: ({ row }) => (
      <span className="block max-w-56 truncate font-mono text-xs" title={row.original.access_group}>
        {row.original.access_group}
      </span>
    ),
  },
  {
    id: "models",
    meta: { title: "Models", skeleton: "chips" },
    header: "Models",
    size: 280,
    enableSorting: false,
    cell: ({ row }) => <ModelsCell models={row.original.model_names} />,
  },
  {
    id: "deployment_count",
    accessorKey: "deployment_count",
    meta: { title: "Deployments", numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title="Deployments" />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => row.original.deployment_count,
  },
  {
    id: "spend",
    accessorKey: "spend",
    meta: { title: "Shared Spend" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Shared Spend" />,
    size: 180,
    enableSorting: true,
    cell: ({ row }) => (
      <SpendBudgetCell spend={row.original.spend} maxBudget={row.original.budget?.max_budget} budgetDecimals={2} />
    ),
  },
  {
    id: "budget_duration",
    meta: { title: "Resets" },
    header: "Resets",
    size: 110,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="text-sm text-muted-foreground">
        {getBudgetDurationLabel(row.original.budget?.budget_duration)}
      </span>
    ),
  },
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">Actions</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) => (
      <div className="flex justify-end">
        <AccessGroupRowActions accessGroup={row.original} onSetBudget={onSetBudget} onClearBudget={onClearBudget} />
      </div>
    ),
  },
];
