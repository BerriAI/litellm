import React, { useMemo, useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow, Icon, Button, Badge } from "@tremor/react";
import { TrashIcon, PencilIcon, SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon } from "@heroicons/react/outline";
import { Tooltip, Tag } from "antd";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { Policy } from "./types";

/** One row per policy name; primaryPolicy is used for display and for Edit (FlowBuilder loads all versions) */
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
    // Prefer production, then highest version_number
    const primary =
      versions.find((v) => v.version_status === "production") ??
      [...versions].sort((a, b) => (b.version_number ?? 0) - (a.version_number ?? 0))[0] ??
      versions[0];
    rows.push({ policy_name: policyName, primaryPolicy: primary, versionCount: versions.length });
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

const PolicyTable: React.FC<PolicyTableProps> = ({
  policies,
  isLoading,
  onDeleteClick,
  onEditClick,
  onViewClick,
  isAdmin = false,
}) => {
  const [sorting, setSorting] = useState<SortingState>([{ id: "policy_name", desc: false }]);

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
            <Tooltip title={`${primaryPolicy.policy_name || "-"}${versionCount > 1 ? ` (${versionCount} versions)` : ""}`}>
              <Button
                size="xs"
                variant="light"
                className="font-medium text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left"
                onClick={() => primaryPolicy.policy_id && onViewClick(primaryPolicy.policy_id)}
              >
                {primaryPolicy.policy_name || "-"}
              </Button>
            </Tooltip>
            {versionCount > 1 && (
              <Badge color="gray" size="xs">
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
          <Tooltip title={policy.description}>
            <span className="text-xs truncate max-w-[200px] block">
              {policy.description || "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      header: "Inherits From",
      accessorFn: (row) => row.primaryPolicy.inherit ?? "",
      cell: ({ row }) => {
        const policy = row.original.primaryPolicy;
        return policy.inherit ? (
          <Badge color="blue" size="xs">
            {policy.inherit}
          </Badge>
        ) : (
          <span className="text-xs text-gray-400">-</span>
        );
      },
    },
    {
      header: "Guardrails (Add)",
      accessorFn: (row) => (row.primaryPolicy.guardrails_add ?? []).join(", "),
      cell: ({ row }) => {
        const policy = row.original.primaryPolicy;
        const guardrails = policy.guardrails_add || [];
        if (guardrails.length === 0) {
          return <span className="text-xs text-gray-400">-</span>;
        }
        return (
          <div className="flex flex-wrap gap-1">
            {guardrails.slice(0, 2).map((g, i) => (
              <Tag key={i} color="green" className="text-xs">
                {g}
              </Tag>
            ))}
            {guardrails.length > 2 && (
              <Tooltip title={guardrails.slice(2).join(", ")}>
                <Tag className="text-xs">+{guardrails.length - 2}</Tag>
              </Tooltip>
            )}
          </div>
        );
      },
    },
    {
      header: "Guardrails (Remove)",
      accessorFn: (row) => (row.primaryPolicy.guardrails_remove ?? []).join(", "),
      cell: ({ row }) => {
        const policy = row.original.primaryPolicy;
        const guardrails = policy.guardrails_remove || [];
        if (guardrails.length === 0) {
          return <span className="text-xs text-gray-400">-</span>;
        }
        return (
          <div className="flex flex-wrap gap-1">
            {guardrails.slice(0, 2).map((g, i) => (
              <Tag key={i} color="red" className="text-xs">
                {g}
              </Tag>
            ))}
            {guardrails.length > 2 && (
              <Tooltip title={guardrails.slice(2).join(", ")}>
                <Tag className="text-xs">+{guardrails.length - 2}</Tag>
              </Tooltip>
            )}
          </div>
        );
      },
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
          return <span className="text-xs text-gray-400">-</span>;
        }
        return (
          <Tooltip title={typeof modelCondition === "string" ? modelCondition : JSON.stringify(modelCondition)}>
            <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">
              {typeof modelCondition === "string"
                ? modelCondition.length > 20
                  ? modelCondition.slice(0, 20) + "..."
                  : modelCondition
                : "Multiple"}
            </code>
          </Tooltip>
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
          <Tooltip title={policy.created_at}>
            <span className="text-xs">{formatDate(policy.created_at)}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => {
        const { primaryPolicy } = row.original;
        const policy = primaryPolicy;
        return (
          <div className="flex space-x-2">
            {isAdmin && (
              <>
                <Tooltip title="Edit policy">
                  <Icon
                    icon={PencilIcon}
                    size="sm"
                    onClick={() => onEditClick(policy)}
                    className="cursor-pointer hover:text-blue-500"
                  />
                </Tooltip>
                <Tooltip title="Delete policy">
                  <Icon
                    icon={TrashIcon}
                    size="sm"
                    onClick={() =>
                      policy.policy_id &&
                      onDeleteClick(policy.policy_id, policy.policy_name || "Unnamed Policy")
                    }
                    className="cursor-pointer hover:text-red-500"
                  />
                </Tooltip>
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
    state: {
      sorting,
    },
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
                      header.id === "actions" ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]" : ""
                    }`}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center">
                        {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                      </div>
                      {header.id !== "actions" && (
                        <div className="w-4">
                          {header.column.getIsSorted() ? (
                            {
                              asc: <ChevronUpIcon className="h-4 w-4 text-blue-500" />,
                              desc: <ChevronDownIcon className="h-4 w-4 text-blue-500" />,
                            }[header.column.getIsSorted() as string]
                          ) : (
                            <SwitchVerticalIcon className="h-4 w-4 text-gray-400" />
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
                <TableCell colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
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
                          ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                          : ""
                      }`}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
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
