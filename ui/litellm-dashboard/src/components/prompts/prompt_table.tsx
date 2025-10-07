import React, { useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow, Button } from "@tremor/react";
import { SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon, TrashIcon } from "@heroicons/react/outline";
import { Tooltip } from "antd";
import { PromptSpec } from "@/components/networking";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";

interface PromptTableProps {
  promptsList: PromptSpec[];
  isLoading: boolean;
  onPromptClick?: (id: string) => void;
  onDeleteClick?: (id: string, name: string) => void;
  accessToken: string | null;
  isAdmin: boolean;
}

const PromptTable: React.FC<PromptTableProps> = ({
  promptsList,
  isLoading,
  onPromptClick,
  onDeleteClick,
  accessToken,
  isAdmin,
}) => {
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);

  // Format date helper function
  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const columns: ColumnDef<PromptSpec>[] = [
    {
      header: "Prompt ID",
      accessorKey: "prompt_id",
      cell: (info: any) => (
        <Tooltip title={String(info.getValue() || "")}>
          <Button
            size="xs"
            variant="light"
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
            onClick={() => info.getValue() && onPromptClick?.(info.getValue())}
          >
            {info.getValue() ? `${String(info.getValue()).slice(0, 7)}...` : ""}
          </Button>
        </Tooltip>
      ),
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      cell: ({ row }) => {
        const prompt = row.original;
        return (
          <Tooltip title={prompt.created_at}>
            <span className="text-xs">{formatDate(prompt.created_at)}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Updated At",
      accessorKey: "updated_at",
      cell: ({ row }) => {
        const prompt = row.original;
        return (
          <Tooltip title={prompt.updated_at}>
            <span className="text-xs">{formatDate(prompt.updated_at)}</span>
          </Tooltip>
        );
      },
    },
    {
      header: "Type",
      accessorKey: "prompt_info.prompt_type",
      cell: ({ row }) => {
        const prompt = row.original;
        return (
          <Tooltip title={prompt.prompt_info.prompt_type}>
            <span className="text-xs">{prompt.prompt_info.prompt_type}</span>
          </Tooltip>
        );
      },
    },
    ...(isAdmin
      ? [
          {
            header: "Actions",
            id: "actions",
            enableSorting: false,
            cell: ({ row }: any) => {
              const prompt = row.original;
              const promptName = prompt.prompt_id || "Unknown Prompt";

              return (
                <div className="flex items-center gap-1">
                  <Tooltip title="Delete prompt">
                    <Button
                      size="xs"
                      variant="light"
                      color="red"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteClick?.(prompt.prompt_id, promptName);
                      }}
                      icon={TrashIcon}
                      className="text-red-500 hover:text-red-700 hover:bg-red-50"
                    />
                  </Tooltip>
                </div>
              );
            },
          },
        ]
      : []),
  ];

  const table = useReactTable({
    data: promptsList,
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
                    className="py-1 h-8"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center">
                        {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                      </div>
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
            ) : promptsList.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-8 text-center">
                  <div className="text-center text-gray-500">
                    <p>No prompts found</p>
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

export default PromptTable;
