import { ChevronDownIcon, ChevronUpIcon, SwitchVerticalIcon } from "@heroicons/react/outline";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow } from "@tremor/react";
import { Tooltip } from "antd";
import React from "react";
import TableIconActionButton from "../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { getProviderLogoAndName } from "../provider_info_helpers";
import { VectorStore } from "./types";

interface VectorStoreTableProps {
  data: VectorStore[];
  onView: (vectorStoreId: string) => void;
  onEdit: (vectorStoreId: string) => void;
  onDelete: (vectorStoreId: string) => void;
}

const VectorStoreTable: React.FC<VectorStoreTableProps> = ({ data, onView, onEdit, onDelete }) => {
  const [sorting, setSorting] = React.useState<SortingState>([{ id: "created_at", desc: true }]);

  const columns: ColumnDef<VectorStore>[] = [
    {
      header: "Vector Store ID",
      accessorKey: "vector_store_id",
      cell: ({ row }) => {
        const vectorStore = row.original;
        return (
          <button
            onClick={() => onView(vectorStore.vector_store_id)}
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[15ch]"
          >
            {vectorStore.vector_store_id.length > 15
              ? `${vectorStore.vector_store_id.slice(0, 15)}...`
              : vectorStore.vector_store_id}
          </button>
        );
      },
    },
    {
      header: "Name",
      accessorKey: "vector_store_name",
      cell: ({ row }) => {
        const vectorStore = row.original;
        return (
          <Tooltip title={vectorStore.vector_store_name}>
            <span className="text-xs">{vectorStore.vector_store_name || "-"}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Description",
      accessorKey: "vector_store_description",
      cell: ({ row }) => {
        const vectorStore = row.original;
        return (
          <Tooltip title={vectorStore.vector_store_description}>
            <span className="text-xs">{vectorStore.vector_store_description || "-"}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Files",
      accessorKey: "vector_store_metadata",
      cell: ({ row }) => {
        const vectorStore = row.original;
        const ingestedFiles = vectorStore.vector_store_metadata?.ingested_files || [];
        
        if (ingestedFiles.length === 0) {
          return <span className="text-xs text-gray-400">-</span>;
        }
        
        const filenames = ingestedFiles
          .map((file) => file.filename || file.file_url || "Unknown")
          .join(", ");
        
        const displayText = ingestedFiles.length === 1 
          ? ingestedFiles[0].filename || ingestedFiles[0].file_url || "1 file"
          : `${ingestedFiles.length} files`;
        
        return (
          <Tooltip title={filenames}>
            <span className="text-xs text-blue-600">{displayText}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Provider",
      accessorKey: "custom_llm_provider",
      cell: ({ row }) => {
        const vectorStore = row.original;
        const { displayName, logo } = getProviderLogoAndName(vectorStore.custom_llm_provider);
        return (
          <div className="flex items-center space-x-2">
            {logo && <img src={logo} alt={displayName} className="h-4 w-4" />}
            <span className="text-xs">{displayName}</span>
          </div>
        );
      },
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      sortingFn: "datetime",
      cell: ({ row }) => {
        const vectorStore = row.original;
        return <span className="text-xs">{new Date(vectorStore.created_at).toLocaleDateString()}</span>;
      },
    },
    {
      header: "Updated At",
      accessorKey: "updated_at",
      sortingFn: "datetime",
      cell: ({ row }) => {
        const vectorStore = row.original;
        return <span className="text-xs">{new Date(vectorStore.updated_at).toLocaleDateString()}</span>;
      },
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => {
        const vectorStore = row.original;
        return (
          <div className="flex space-x-2">
            <TableIconActionButton
              variant="Edit"
              tooltipText="Edit vector store"
              onClick={() => onEdit(vectorStore.vector_store_id)}
            />
            <TableIconActionButton
              variant="Delete"
              tooltipText="Delete vector store"
              onClick={() => onDelete(vectorStore.vector_store_id)}
            />
          </div>
        );
      },
    },
  ];

  const table = useReactTable({
    data,
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
            {table.getRowModel().rows.length > 0 ? (
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
                    <p>No vector stores found</p>
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

export default VectorStoreTable;
