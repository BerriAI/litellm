import React, { useMemo, useState } from "react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Pencil,
  Trash2,
} from "lucide-react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { Policy } from "./types";

interface PolicyRow {
  policy_name: string;
  primaryPolicy: Policy;
  versionCount: number;
}

function groupPoliciesByName(policies: Policy[]): PolicyRow[] {
  const byName = new Map<string, Policy[]>();
  for (const p of policies) {
    const name = p.policy_name || "(unnamed)";
    if (!byName.has(name)) byName.set(name, []);
    byName.get(name)!.push(p);
  }
  const rows: PolicyRow[] = [];
  for (const [policyName, versions] of byName) {
    const primary =
      versions.find((v) => v.version_status === "production") ??
      [...versions].sort(
        (a, b) => (b.version_number ?? 0) - (a.version_number ?? 0),
      )[0] ??
      versions[0];
    rows.push({
      policy_name: policyName,
      primaryPolicy: primary,
      versionCount: versions.length,
    });
  }
  return rows.sort((a, b) => a.policy_name.localeCompare(b.policy_name));
}

interface PolicyTableProps {
  policies: Policy[];
  isLoading: boolean;
  onDeleteClick: (policyId: string, policyName: string) => void;
  onEditClick: (policy: Policy) => void;
  onViewClick: (policyId: string) => void;
  isAdmin?: boolean;
}

const chipList = (items: string[], classes: string) => {
  if (items.length === 0)
    return <span className="text-xs text-muted-foreground">-</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {items.slice(0, 2).map((g, i) => (
        <Badge key={i} className={`text-xs ${classes}`}>
          {g}
        </Badge>
      ))}
      {items.length > 2 && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge className="text-xs bg-muted text-muted-foreground">
                +{items.length - 2}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>{items.slice(2).join(", ")}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
};

const PolicyTable: React.FC<PolicyTableProps> = ({
  policies,
  isLoading,
  onDeleteClick,
  onEditClick,
  onViewClick,
  isAdmin = false,
}) => {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "policy_name", desc: false },
  ]);

  const rows = useMemo(() => groupPoliciesByName(policies), [policies]);

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const columns: ColumnDef<PolicyRow>[] = [
    {
      header: "Name",
      accessorKey: "policy_name",
      cell: ({ row }) => {
        const { primaryPolicy, versionCount } = row.original;
        return (
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    className="font-medium text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-xs px-2 py-0.5 text-left rounded"
                    onClick={() =>
                      primaryPolicy.policy_id &&
                      onViewClick(primaryPolicy.policy_id)
                    }
                  >
                    {primaryPolicy.policy_name || "-"}
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  {primaryPolicy.policy_name || "-"}
                  {versionCount > 1 ? ` (${versionCount} versions)` : ""}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            {versionCount > 1 && (
              <Badge className="text-xs bg-muted text-muted-foreground">
                {versionCount} version{versionCount !== 1 ? "s" : ""}
              </Badge>
            )}
          </div>
        );
      },
    },
    {
      header: "Description",
      accessorFn: (row) => row.primaryPolicy.description ?? "",
      cell: ({ row }) => {
        const policy = row.original.primaryPolicy;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs truncate max-w-[200px] block">
                  {policy.description || "-"}
                </span>
              </TooltipTrigger>
              <TooltipContent>{policy.description || "-"}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Inherits From",
      accessorFn: (row) => row.primaryPolicy.inherit ?? "",
      cell: ({ row }) => {
        const policy = row.original.primaryPolicy;
        return policy.inherit ? (
          <Badge className="text-xs bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
            {policy.inherit}
          </Badge>
        ) : (
          <span className="text-xs text-muted-foreground">-</span>
        );
      },
    },
    {
      header: "Guardrails (Add)",
      accessorFn: (row) => (row.primaryPolicy.guardrails_add ?? []).join(", "),
      cell: ({ row }) =>
        chipList(
          row.original.primaryPolicy.guardrails_add || [],
          "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
        ),
    },
    {
      header: "Guardrails (Remove)",
      accessorFn: (row) =>
        (row.primaryPolicy.guardrails_remove ?? []).join(", "),
      cell: ({ row }) =>
        chipList(
          row.original.primaryPolicy.guardrails_remove || [],
          "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
        ),
    },
    {
      header: "Model Condition",
      accessorFn: (row) => {
        const m = row.primaryPolicy.condition?.model;
        return typeof m === "string" ? m : JSON.stringify(m ?? "");
      },
      cell: ({ row }) => {
        const policy = row.original.primaryPolicy;
        const modelCondition = policy.condition?.model;
        if (!modelCondition) {
          return <span className="text-xs text-muted-foreground">-</span>;
        }
        const asString =
          typeof modelCondition === "string"
            ? modelCondition
            : JSON.stringify(modelCondition);
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <code className="text-xs bg-muted px-1 py-0.5 rounded">
                  {typeof modelCondition === "string"
                    ? modelCondition.length > 20
                      ? modelCondition.slice(0, 20) + "..."
                      : modelCondition
                    : "Multiple"}
                </code>
              </TooltipTrigger>
              <TooltipContent>{asString}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      header: "Created At",
      id: "created_at",
      accessorFn: (row) => row.primaryPolicy.created_at ?? "",
      cell: ({ row }) => {
        const policy = row.original.primaryPolicy;
        return (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">{formatDate(policy.created_at)}</span>
              </TooltipTrigger>
              <TooltipContent>{policy.created_at}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      },
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => {
        const policy = row.original.primaryPolicy;
        return (
          <div className="flex space-x-2 items-center">
            {isAdmin && (
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-primary"
                        onClick={() => onEditClick(policy)}
                        aria-label="Edit policy"
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Edit policy</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-destructive"
                        onClick={() =>
                          policy.policy_id &&
                          onDeleteClick(
                            policy.policy_id,
                            policy.policy_name || "Unnamed Policy",
                          )
                        }
                        aria-label="Delete policy"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Delete policy</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </>
            )}
          </div>
        );
      },
    },
  ];

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
  });

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <Table className="[&_td]:py-0.5 [&_th]:py-1">
          <TableHead>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHeaderCell
                    key={header.id}
                    className={`py-1 h-8 ${
                      header.id === "actions"
                        ? "sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                        : ""
                    }`}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center">
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext(),
                            )}
                      </div>
                      {header.id !== "actions" && (
                        <div className="w-4">
                          {header.column.getIsSorted() ? (
                            {
                              asc: (
                                <ChevronUp className="h-4 w-4 text-primary" />
                              ),
                              desc: (
                                <ChevronDown className="h-4 w-4 text-primary" />
                              ),
                            }[header.column.getIsSorted() as string]
                          ) : (
                            <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
                          )}
                        </div>
                      )}
                    </div>
                  </TableHeaderCell>
                ))}
              </TableRow>
            ))}
          </TableHead>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-8 text-center"
                >
                  <div className="text-center text-muted-foreground">
                    <p>Loading...</p>
                  </div>
                </TableCell>
              </TableRow>
            ) : rows.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.original.policy_name} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${
                        cell.column.id === "actions"
                          ? "sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                          : ""
                      }`}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-8 text-center"
                >
                  <div className="text-center text-muted-foreground">
                    <p>No policies found</p>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};

export default PolicyTable;
