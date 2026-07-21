"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Tooltip } from "antd";

import { ToolRow } from "@/components/networking";
import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdCell, IdentityCell } from "@/components/shared/table_cells";

import { PolicySelect } from "./PolicySelect";

interface ToolPoliciesTableColumnsDeps {
  onSelectTool: (toolName: string) => void;
  savingInput: string | null;
  savingOutput: string | null;
  onInputPolicyChange: (toolName: string, policy: string) => void;
  onOutputPolicyChange: (toolName: string, policy: string) => void;
}

function TruncatedText({ value, className }: { value: string | undefined; className?: string }) {
  const text = value ?? "-";
  return (
    <Tooltip title={text}>
      <span className={className}>{text}</span>
    </Tooltip>
  );
}

export const getToolPoliciesTableColumns = ({
  onSelectTool,
  savingInput,
  savingOutput,
  onInputPolicyChange,
  onOutputPolicyChange,
}: ToolPoliciesTableColumnsDeps): ColumnDef<ToolRow>[] => [
  {
    id: "created_at",
    accessorFn: (row) => row.created_at ?? "",
    header: ({ column }) => <DataTableSortHeader column={column} title="Discovered" />,
    size: 170,
    enableGlobalFilter: false,
    cell: ({ row }) => <DateCell value={row.original.created_at} />,
  },
  {
    id: "tool_name",
    accessorFn: (row) => row.tool_name,
    header: ({ column }) => <DataTableSortHeader column={column} title="Tool Name" />,
    minSize: 200,
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.tool_name}
        titleClassName="font-mono text-xs font-normal text-primary"
        className="max-w-60"
        onClick={() => onSelectTool(row.original.tool_name)}
      />
    ),
  },
  {
    id: "input_policy",
    accessorFn: (row) => row.input_policy,
    header: ({ column }) => <DataTableSortHeader column={column} title="Input Policy" />,
    size: 140,
    filterFn: "equalsString",
    meta: { title: "Input Policy", skeleton: "badge" },
    cell: ({ row }) => (
      <PolicySelect
        value={row.original.input_policy}
        toolName={row.original.tool_name}
        saving={savingInput === row.original.tool_name}
        onChange={onInputPolicyChange}
        policyType="input"
      />
    ),
  },
  {
    id: "output_policy",
    accessorFn: (row) => row.output_policy,
    header: ({ column }) => <DataTableSortHeader column={column} title="Output Policy" />,
    size: 140,
    filterFn: "equalsString",
    meta: { title: "Output Policy", skeleton: "badge" },
    cell: ({ row }) => (
      <PolicySelect
        value={row.original.output_policy}
        toolName={row.original.tool_name}
        saving={savingOutput === row.original.tool_name}
        onChange={onOutputPolicyChange}
        policyType="output"
      />
    ),
  },
  {
    id: "call_count",
    accessorFn: (row) => row.call_count ?? 0,
    header: ({ column }) => <DataTableSortHeader column={column} title="# Calls" />,
    size: 100,
    enableGlobalFilter: false,
    meta: { numeric: true },
    cell: ({ row }) => <span className="font-mono">{(row.original.call_count ?? 0).toLocaleString()}</span>,
  },
  {
    id: "team_id",
    accessorFn: (row) => row.team_id ?? "",
    header: ({ column }) => <DataTableSortHeader column={column} title="Team Name" />,
    size: 160,
    filterFn: "equalsString",
    meta: { title: "Team Name" },
    cell: ({ row }) => <IdCell value={row.original.team_id} variant="plain" />,
  },
  {
    id: "key_hash",
    accessorFn: (row) => row.key_hash ?? "",
    header: "Key Hash",
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.key_hash} />,
  },
  {
    id: "key_alias",
    accessorFn: (row) => row.key_alias ?? "",
    header: ({ column }) => <DataTableSortHeader column={column} title="Key Name" />,
    size: 150,
    filterFn: "equalsString",
    meta: { title: "Key Name" },
    cell: ({ row }) => <TruncatedText value={row.original.key_alias} className="block max-w-32 truncate" />,
  },
  {
    id: "user_agent",
    accessorFn: (row) => row.user_agent ?? "",
    header: "User Agent",
    size: 180,
    enableSorting: false,
    enableGlobalFilter: false,
    cell: ({ row }) => (
      <TruncatedText
        value={row.original.user_agent}
        className="block max-w-40 truncate font-mono text-muted-foreground"
      />
    ),
  },
];
