"use client";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { ArrowUpDown, ChevronDown, ChevronUp } from "lucide-react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  PaginationState,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import React, { useState } from "react";
import { KeyResponse } from "../../key_team_helpers/key_list";

const TipSpan: React.FC<{
  value: string | null | undefined;
  className: string;
}> = ({ value, className }) => {
  if (!value) return <span className={className}>-</span>;
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={className}>{value}</span>
        </TooltipTrigger>
        <TooltipContent>{value}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

interface DeletedKeysTableProps {
  keys: KeyResponse[];
  totalCount: number;
  isLoading: boolean;
  isFetching: boolean;
  pageIndex: number;
  pageSize: number;
  onPageChange: (pageIndex: number) => void;
}

export function DeletedKeysTable({
  keys,
  totalCount,
  isLoading,
  isFetching,
  pageIndex,
  pageSize,
  onPageChange,
}: DeletedKeysTableProps) {
  const [sorting, setSorting] = useState<SortingState>([
    {
      id: "deleted_at",
      desc: true,
    },
  ]);

  const [tablePagination, setTablePagination] = useState<PaginationState>({
    pageIndex,
    pageSize,
  });

  // Sync pagination state when prop changes
  React.useEffect(() => {
    setTablePagination({ pageIndex, pageSize });
  }, [pageIndex, pageSize]);

  const columns: ColumnDef<KeyResponse>[] = [
    {
      id: "token",
      accessorKey: "token",
      header: "Key ID",
      size: 150,
      maxSize: 250,
      cell: (info) => (
        <TipSpan
          value={info.getValue() as string}
          className="font-mono text-blue-500 text-xs truncate block max-w-[250px]"
        />
      ),
    },
    {
      id: "key_alias",
      accessorKey: "key_alias",
      header: "Key Alias",
      size: 150,
      maxSize: 200,
      cell: (info) => (
        <TipSpan
          value={info.getValue() as string}
          className="font-mono text-xs truncate block max-w-[200px]"
        />
      ),
    },
    {
      id: "team_alias",
      accessorKey: "team_alias",
      header: "Team Alias",
      size: 120,
      maxSize: 180,
      cell: (info) => {
        const value = info.getValue() as string;
        return (
          <span className="truncate block max-w-[180px]">
            {value || "-"}
          </span>
        );
      },
    },
    {
      id: "spend",
      accessorKey: "spend",
      header: "Spend (USD)",
      size: 100,
      maxSize: 140,
      cell: (info) => (
        <span className="block max-w-[140px]">
          {formatNumberWithCommas(info.getValue() as number, 4)}
        </span>
      ),
    },
    {
      id: "max_budget",
      accessorKey: "max_budget",
      header: "Budget (USD)",
      size: 110,
      maxSize: 150,
      cell: (info) => {
        const maxBudget = info.getValue() as number | null;
        return (
          <span className="block max-w-[150px]">
            {maxBudget === null ? "Unlimited" : `$${formatNumberWithCommas(maxBudget)}`}
          </span>
        );
      },
    },
    {
      id: "user_email",
      accessorKey: "user_email",
      header: "User Email",
      size: 160,
      maxSize: 250,
      cell: (info) => (
        <TipSpan
          value={info.getValue() as string}
          className="font-mono text-xs truncate block max-w-[250px]"
        />
      ),
    },
    {
      id: "user_id",
      accessorKey: "user_id",
      header: "User ID",
      size: 120,
      maxSize: 200,
      cell: (info) => (
        <TipSpan
          value={info.getValue() as string | null}
          className="truncate block max-w-[200px]"
        />
      ),
    },
    {
      id: "created_at",
      accessorKey: "created_at",
      header: "Created At",
      size: 120,
      maxSize: 140,
      cell: (info) => {
        const value = info.getValue();
        return (
          <span className="block max-w-[140px]">
            {value ? new Date(value as string).toLocaleDateString() : "-"}
          </span>
        );
      },
    },
    {
      id: "created_by",
      accessorKey: "created_by",
      header: "Created By",
      size: 120,
      maxSize: 180,
      cell: (info) => {
        const value = (info.row.original as { created_by?: string | null })
          .created_by;
        return (
          <TipSpan
            value={value}
            className="truncate block max-w-[180px]"
          />
        );
      },
    },
    {
      id: "deleted_at",
      accessorKey: "deleted_at",
      header: "Deleted At",
      size: 120,
      maxSize: 140,
      cell: (info) => {
        const value = (info.row.original as { deleted_at?: string | null })
          .deleted_at;
        return (
          <span className="block max-w-[140px]">
            {value ? new Date(value).toLocaleDateString() : "-"}
          </span>
        );
      },
    },
    {
      id: "deleted_by",
      accessorKey: "deleted_by",
      header: "Deleted By",
      size: 120,
      maxSize: 180,
      cell: (info) => {
        const value = (info.row.original as { deleted_by?: string | null })
          .deleted_by;
        return (
          <TipSpan
            value={value}
            className="truncate block max-w-[180px]"
          />
        );
      },
    },
  ];

  const table = useReactTable({
    data: keys,
    columns,
    columnResizeMode: "onChange",
    columnResizeDirection: "ltr",
    state: {
      sorting,
      pagination: tablePagination,
    },
    onSortingChange: setSorting,
    onPaginationChange: (updater) => {
      const newPagination = typeof updater === "function" ? updater(tablePagination) : updater;
      setTablePagination(newPagination);
      onPageChange(newPagination.pageIndex);
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    enableSorting: true,
    manualSorting: false,
    manualPagination: true,
    pageCount: Math.ceil(totalCount / pageSize),
  });

  const { pageIndex: currentPageIndex } = table.getState().pagination;
  const start = currentPageIndex * pageSize + 1;
  const end = Math.min((currentPageIndex + 1) * pageSize, totalCount);
  const rangeLabel = `${start} - ${end}`;

  return (
    <div className="w-full h-full overflow-hidden">
      <div className="border-b py-4 flex-1 overflow-hidden">
        <div className="flex items-center justify-between w-full mb-4">
          {isLoading || isFetching ? (
            <span className="inline-flex text-sm text-foreground">
              Loading...
            </span>
          ) : (
            <span className="inline-flex text-sm text-foreground">
              Showing {rangeLabel} of {totalCount} results
            </span>
          )}

          <div className="inline-flex items-center gap-2">
            {isLoading || isFetching ? (
              <span className="text-sm text-foreground">Loading...</span>
            ) : (
              <span className="text-sm text-foreground">
                Page {currentPageIndex + 1} of {table.getPageCount()}
              </span>
            )}

            <button
              type="button"
              onClick={() => table.previousPage()}
              disabled={isLoading || isFetching || !table.getCanPreviousPage()}
              className="px-3 py-1 text-sm border border-border rounded-md hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>

            <button
              type="button"
              onClick={() => table.nextPage()}
              disabled={isLoading || isFetching || !table.getCanNextPage()}
              className="px-3 py-1 text-sm border border-border rounded-md hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
        <div className="h-[75vh] overflow-auto">
          <div className="rounded-lg custom-border relative">
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-0.5 [&_th]:py-1" style={{ width: table.getCenterTotalSize() }}>
                <TableHeader>
                  {table.getHeaderGroups().map((headerGroup) => (
                    <TableRow key={headerGroup.id}>
                      {headerGroup.headers.map((header) => (
                        <TableHead
                          key={header.id}
                          data-header-id={header.id}
                          className="py-1 h-8 relative hover:bg-muted"
                          style={{
                            width: header.getSize(),
                            maxWidth: header.column.columnDef.maxSize,
                            position: "relative",
                          }}
                          onMouseEnter={() => {
                            const resizer = document.querySelector(`[data-header-id="${header.id}"] .resizer`);
                            if (resizer) {
                              (resizer as HTMLElement).style.opacity = "0.5";
                            }
                          }}
                          onMouseLeave={() => {
                            const resizer = document.querySelector(`[data-header-id="${header.id}"] .resizer`);
                            if (resizer && !header.column.getIsResizing()) {
                              (resizer as HTMLElement).style.opacity = "0";
                            }
                          }}
                          onClick={header.column.getToggleSortingHandler()}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center">
                              {header.isPlaceholder
                                ? null
                                : flexRender(header.column.columnDef.header, header.getContext())}
                            </div>
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
                            <div
                              onDoubleClick={() => header.column.resetSize()}
                              onMouseDown={header.getResizeHandler()}
                              onTouchStart={header.getResizeHandler()}
                              className={`resizer ${table.options.columnResizeDirection} ${header.column.getIsResizing() ? "isResizing" : ""}`}
                              style={{
                                position: "absolute",
                                right: 0,
                                top: 0,
                                height: "100%",
                                width: "5px",
                                background: header.column.getIsResizing() ? "#3b82f6" : "transparent",
                                cursor: "col-resize",
                                userSelect: "none",
                                touchAction: "none",
                                opacity: header.column.getIsResizing() ? 1 : 0,
                              }}
                            />
                          </div>
                        </TableHead>
                      ))}
                    </TableRow>
                  ))}
                </TableHeader>
                <TableBody>
                  {isLoading || isFetching ? (
                    <TableRow>
                      <TableCell
                        colSpan={columns.length}
                        className="h-8 text-center"
                      >
                        <div className="text-center text-muted-foreground">
                          <p>🚅 Loading keys...</p>
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : keys.length > 0 ? (
                    table.getRowModel().rows.map((row) => (
                      <TableRow key={row.id} className="h-8">
                        {row.getVisibleCells().map((cell) => (
                          <TableCell
                            key={cell.id}
                            style={{
                              width: cell.column.getSize(),
                              maxWidth: cell.column.columnDef.maxSize,
                              whiteSpace: "pre-wrap",
                              overflow: "hidden",
                            }}
                            className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap"
                          >
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
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
                          <p>No deleted keys found</p>
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
