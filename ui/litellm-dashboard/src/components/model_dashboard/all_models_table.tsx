import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  SortingState,
  useReactTable,
  ColumnResizeMode,
  VisibilityState,
  PaginationState,
  OnChangeFn,
} from "@tanstack/react-table";
import React from "react";
import { Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell } from "@tremor/react";
import { TableHeaderSortDropdown, SortState } from "../common_components/TableHeaderSortDropdown/TableHeaderSortDropdown";

// Extend the column meta type to include className
declare module "@tanstack/react-table" {
  interface ColumnMeta<TData, TValue> {
    className?: string;
  }
}

interface AllModelsDataTableProps<TData, TValue> {
  data: TData[];
  columns: ColumnDef<TData, TValue>[];
  isLoading?: boolean;
  sorting?: SortingState;
  onSortingChange?: OnChangeFn<SortingState>;
  pagination?: PaginationState;
  onPaginationChange?: OnChangeFn<PaginationState>;
  enablePagination?: boolean;
}

export function AllModelsDataTable<TData, TValue>({
  data = [],
  columns,
  isLoading = false,
  sorting = [],
  onSortingChange,
  pagination,
  onPaginationChange,
  enablePagination = false,
}: AllModelsDataTableProps<TData, TValue>) {
  const [columnResizeMode] = React.useState<ColumnResizeMode>("onChange");
  const [columnSizing, setColumnSizing] = React.useState({});
  const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({});

  const tableInstance = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnSizing,
      columnVisibility,
      ...(enablePagination && pagination ? { pagination } : {}),
    },
    columnResizeMode,
    onSortingChange: onSortingChange,
    onColumnSizingChange: setColumnSizing,
    onColumnVisibilityChange: setColumnVisibility,
    ...(enablePagination && onPaginationChange ? { onPaginationChange } : {}),
    getCoreRowModel: getCoreRowModel(),
    // NO getSortedRowModel - sorting is handled server-side
    ...(enablePagination ? { getPaginationRowModel: getPaginationRowModel() } : {}),
    enableSorting: true,
    enableColumnResizing: true,
    manualSorting: true, // Enable manual sorting for server-side sorting
    defaultColumn: {
      minSize: 40,
      maxSize: 500,
    },
  });

  const getHeaderText = (header: any): string => {
    if (typeof header === "string") {
      return header;
    }
    if (typeof header === "function") {
      const headerElement = header();
      if (headerElement && headerElement.props && headerElement.props.children) {
        const children = headerElement.props.children;
        if (typeof children === "string") {
          return children;
        }
        if (children.props && children.props.children) {
          return children.props.children;
        }
      }
    }
    return "";
  };

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <div className="relative min-w-full">
          <Table
            className="[&_td]:py-2 [&_th]:py-2"
            style={{
              width: tableInstance.getTotalSize(),
              minWidth: "100%",
              tableLayout: "fixed",
            }}
          >
            <TableHead>
              {tableInstance.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <TableHeaderCell
                      key={header.id}
                      className={`py-1 h-8 relative ${header.id === "actions"
                        ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8"
                        : ""
                        } ${header.column.columnDef.meta?.className || ""}`}
                      style={{
                        width: header.id === "actions" ? 120 : header.getSize(),
                        position: header.id === "actions" ? "sticky" : "relative",
                        right: header.id === "actions" ? 0 : "auto",
                      }}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center">
                          {header.isPlaceholder
                            ? null
                            : flexRender(header.column.columnDef.header, header.getContext())}
                        </div>
                        {header.id !== "actions" && header.column.getCanSort() && onSortingChange && (
                          <TableHeaderSortDropdown
                            sortState={
                              header.column.getIsSorted() === false
                                ? false
                                : (header.column.getIsSorted() as SortState)
                            }
                            onSortChange={(newState) => {
                              // Convert SortState to TanStack SortingState
                              // Only allow one column to be sorted at a time
                              if (newState === false) {
                                onSortingChange([]);
                              } else {
                                onSortingChange([
                                  {
                                    id: header.column.id,
                                    desc: newState === "desc",
                                  },
                                ]);
                              }
                            }}
                            columnId={header.column.id}
                          />
                        )}
                      </div>
                      {header.column.getCanResize() && (
                        <div
                          onMouseDown={header.getResizeHandler()}
                          onTouchStart={header.getResizeHandler()}
                          className={`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${header.column.getIsResizing() ? "bg-blue-500" : "hover:bg-blue-200"
                            }`}
                        />
                      )}
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
                      <p>🚅 Loading models...</p>
                    </div>
                  </TableCell>
                </TableRow>
              ) : tableInstance.getRowModel().rows.length > 0 ? (
                tableInstance.getRowModel().rows.map((row) => (
                  <TableRow key={row.id}>
                    {row.getVisibleCells().map((cell) => (
                      <TableCell
                        key={cell.id}
                        className={`py-0.5 overflow-hidden ${cell.column.id === "actions"
                          ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8"
                          : ""
                          } ${cell.column.columnDef.meta?.className || ""}`}
                        style={{
                          width: cell.column.id === "actions" ? 120 : cell.column.getSize(),
                          position: cell.column.id === "actions" ? "sticky" : "relative",
                          right: cell.column.id === "actions" ? 0 : "auto",
                        }}
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
                      <p>No models found</p>
                    </div>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
}
