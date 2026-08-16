import { Fragment, useState } from "react";
import {
  ColumnDef,
  RowData,
  flexRender,
  getCoreRowModel,
  getExpandedRowModel,
  Row,
  useReactTable,
  getSortedRowModel,
  SortingState,
} from "@tanstack/react-table";

import { Table, TableHeader, TableHead, TableBody, TableRow, TableCell } from "@/components/ui/table";

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends RowData, TValue> {
    numeric?: boolean;
  }
}

interface DataTableProps<TData, TValue> {
  data: TData[];
  columns: ColumnDef<TData, TValue>[];
  getRowId?: (row: TData, index: number) => string;
  onRowClick?: (row: TData) => void;
  /** Renders inside a single colspan cell */
  renderSubComponent?: (props: { row: Row<TData> }) => React.ReactElement;
  getRowCanExpand?: (row: Row<TData>) => boolean;
  isLoading?: boolean;
  loadingMessage?: string;
  noDataMessage?: string;
  /** Enable client-side column sorting (defaults to false to avoid conflicts with server-side sorting) */
  enableSorting?: boolean;
}

export function DataTable<TData, TValue>({
  data = [],
  columns,
  getRowId,
  onRowClick,
  renderSubComponent,
  getRowCanExpand,
  isLoading = false,
  loadingMessage = "Loading...",
  noDataMessage = "No results",
  enableSorting = false,
}: DataTableProps<TData, TValue>) {
  const supportsExpansion = !!renderSubComponent && !!getRowCanExpand;
  const hasExplicitColumnSizes = columns.some((column) => column.size !== undefined);
  const [sorting, setSorting] = useState<SortingState>([]);

  const table = useReactTable<TData>({
    data,
    columns,
    ...(enableSorting && {
      state: {
        sorting,
      },
      onSortingChange: setSorting,
      enableSortingRemoval: false,
    }),
    ...(supportsExpansion && { getRowCanExpand }),
    ...(getRowId && { getRowId }),
    getCoreRowModel: getCoreRowModel(),
    ...(enableSorting && { getSortedRowModel: getSortedRowModel() }),
    ...(supportsExpansion && { getExpandedRowModel: getExpandedRowModel() }),
  });

  const tableClassName = hasExplicitColumnSizes ? "table-fixed" : "table-fixed w-full box-border";
  const tableStyle = hasExplicitColumnSizes ? { minWidth: table.getCenterTotalSize() } : { minWidth: "400px" };

  return (
    <div className="rounded-lg custom-border overflow-hidden w-full max-w-full box-border">
      <Table className={tableClassName} style={tableStyle}>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id} className="bg-muted/50 hover:bg-muted/50">
              {headerGroup.headers.map((header) => {
                const canSort = enableSorting && header.column.getCanSort();
                const isSorted = header.column.getIsSorted();
                const numeric = header.column.columnDef.meta?.numeric;

                return (
                  <TableHead
                    key={header.id}
                    className={`py-1 h-8 text-xs font-medium text-muted-foreground first:pl-4 last:pr-4 ${
                      canSort ? "cursor-pointer select-none hover:bg-muted" : ""
                    }`}
                    style={hasExplicitColumnSizes ? { width: header.getSize() } : undefined}
                    onClick={canSort ? header.column.getToggleSortingHandler() : undefined}
                  >
                    {header.isPlaceholder ? null : (
                      <div className={`flex items-center gap-1 ${numeric ? "justify-end" : ""}`}>
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {canSort && (
                          <span className="text-muted-foreground">
                            {isSorted === "asc" ? "↑" : isSorted === "desc" ? "↓" : "⇅"}
                          </span>
                        )}
                      </div>
                    )}
                  </TableHead>
                );
              })}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {isLoading ? (
            <TableRow className="hover:bg-transparent">
              <TableCell colSpan={columns.length} className="h-8 text-center">
                <div className="text-center text-muted-foreground">
                  <p>{loadingMessage}</p>
                </div>
              </TableCell>
            </TableRow>
          ) : table.getRowModel().rows.length > 0 ? (
            table.getRowModel().rows.map((row) => (
              <Fragment key={row.id}>
                <TableRow
                  className={`h-8 ${onRowClick ? "cursor-pointer" : ""}`}
                  onClick={() => onRowClick?.(row.original)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap first:pl-4 last:pr-4 ${
                        cell.column.columnDef.meta?.numeric ? "text-right tabular-nums" : ""
                      }`}
                      style={hasExplicitColumnSizes ? { width: cell.column.getSize() } : undefined}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>

                {supportsExpansion && row.getIsExpanded() && renderSubComponent && (
                  <TableRow className="hover:bg-transparent">
                    <TableCell colSpan={row.getVisibleCells().length} className="p-0">
                      <div className="w-full max-w-full overflow-hidden box-border">{renderSubComponent({ row })}</div>
                    </TableCell>
                  </TableRow>
                )}
              </Fragment>
            ))
          ) : (
            <TableRow className="hover:bg-transparent">
              <TableCell colSpan={columns.length} className="h-24 text-center align-middle">
                <p className="text-sm text-muted-foreground">{noDataMessage}</p>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
