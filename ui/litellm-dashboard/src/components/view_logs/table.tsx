import { Fragment, useState } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getExpandedRowModel,
  Row,
  useReactTable,
  getSortedRowModel,
  SortingState,
} from "@tanstack/react-table";

import { Table, TableHeader, TableHead, TableBody, TableRow, TableCell } from "@/components/ui/table";

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
    <div className="rounded-lg custom-border w-full max-w-full box-border">
      <Table className={tableClassName} style={tableStyle}>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => {
                const canSort = enableSorting && header.column.getCanSort();
                const isSorted = header.column.getIsSorted();

                return (
                  <TableHead
                    key={header.id}
                    className={`py-1 h-8 ${canSort ? "cursor-pointer select-none hover:bg-muted" : ""}`}
                    style={hasExplicitColumnSizes ? { width: header.getSize() } : undefined}
                    onClick={canSort ? header.column.getToggleSortingHandler() : undefined}
                  >
                    {header.isPlaceholder ? null : (
                      <div className="flex items-center gap-1">
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
            <TableRow>
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
                      className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap"
                      style={hasExplicitColumnSizes ? { width: cell.column.getSize() } : undefined}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>

                {supportsExpansion && row.getIsExpanded() && renderSubComponent && (
                  <TableRow>
                    <TableCell colSpan={row.getVisibleCells().length} className="p-0">
                      <div className="w-full max-w-full overflow-hidden box-border">{renderSubComponent({ row })}</div>
                    </TableCell>
                  </TableRow>
                )}
              </Fragment>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={columns.length} className="h-8 text-center">
                <div className="text-center text-muted-foreground">
                  <p>{noDataMessage}</p>
                </div>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
