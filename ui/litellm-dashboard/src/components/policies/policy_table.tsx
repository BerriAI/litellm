import React, { useState } from "react";
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
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);

  // Format date helper function
  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const columns: ColumnDef<Policy>[] = [
    {
      header: "Policy ID",
      accessorKey: "policy_id",
      cell: (info: any) => (
        <Tooltip title={String(info.getValue() || "")}>
          <Button
            size="xs"
            variant="light"
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
            onClick={() => info.getValue() && onViewClick(info.getValue())}
          >
            {info.getValue() ? `${String(info.getValue()).slice(0, 7)}...` : ""}
          </Button>
        </Tooltip>
      ),
    },
    {
      header: "Name",
      accessorKey: "policy_name",
      cell: ({ row }) => {
        const policy = row.original;
        return (
          <Tooltip title={policy.policy_name}>
            <span className="text-xs font-medium">{policy.policy_name || "-"}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Description",
      accessorKey: "description",
      cell: ({ row }) => {
        const policy = row.original;
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
      accessorKey: "inherit",
      cell: ({ row }) => {
        const policy = row.original;
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
      accessorKey: "guardrails_add",
      cell: ({ row }) => {
        const policy = row.original;
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
      accessorKey: "guardrails_remove",
      cell: ({ row }) => {
        const policy = row.original;
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
      accessorKey: "condition",
      cell: ({ row }) => {
        const policy = row.original;
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
      accessorKey: "created_at",
      cell: ({ row }) => {
        const policy = row.original;
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
        const policy = row.original;
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
    data: policies,
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
            ) : policies.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
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
