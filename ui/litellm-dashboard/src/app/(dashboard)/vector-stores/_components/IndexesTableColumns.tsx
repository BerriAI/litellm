"use client";

import { ColumnDef } from "@tanstack/react-table";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell } from "@/components/shared/table_cells";

import type { VectorStoreIndex } from "./IndexesTab";

export const getIndexesTableColumns = (): ColumnDef<VectorStoreIndex>[] => [
  {
    id: "index_name",
    accessorKey: "index_name",
    meta: { title: "Index Name" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Index Name" />,
    size: 220,
    enableSorting: true,
    cell: ({ row }) => (
      <span className="block max-w-60 truncate text-sm font-medium" title={row.original.index_name}>
        {row.original.index_name || "-"}
      </span>
    ),
  },
  {
    id: "vector_store_name",
    accessorFn: (row) => row.litellm_params.vector_store_name,
    meta: { title: "Vector Store" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Vector Store" />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => (
      <span className="block max-w-60 truncate text-sm" title={row.original.litellm_params.vector_store_name}>
        {row.original.litellm_params.vector_store_name || "-"}
      </span>
    ),
  },
  {
    id: "vector_store_index",
    accessorFn: (row) => row.litellm_params.vector_store_index,
    meta: { title: "Provider Index" },
    header: "Provider Index",
    size: 220,
    enableSorting: false,
    cell: ({ row }) => (
      <span
        className="block max-w-60 truncate font-mono text-xs"
        title={row.original.litellm_params.vector_store_index}
      >
        {row.original.litellm_params.vector_store_index || "-"}
      </span>
    ),
  },
  {
    id: "created_by",
    accessorKey: "created_by",
    meta: { title: "Created By" },
    header: "Created By",
    size: 160,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="block max-w-48 truncate text-sm" title={row.original.created_by ?? undefined}>
        {row.original.created_by || "-"}
      </span>
    ),
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    sortingFn: "datetime",
    meta: { title: "Created At" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Created At" />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
];
