import { Fragment, useState } from "react";
import { ColumnDef, flexRender, getCoreRowModel, getExpandedRowModel, Row, useReactTable, getSortedRowModel, SortingState } from "@tanstack/react-table";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface DataTableProps<TData, TValue> {
  data: TData[];
  columns: ColumnDef<TData, TValue>[];
  onRowClick?: (row: TData) => void;
  /** Renders inside a single colspan cell (used by audit logs) */
  renderSubComponent?: (props: { row: Row<TData> }) => React.ReactElement;
  /** Renders directly in tbody as sibling table rows (used by MCP children) */
  renderChildRows?: (props: { row: Row<TData> }) => React.ReactNode;
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
  onRowClick,
  renderSubComponent,
  renderChildRows,
  getRowCanExpand,
  isLoading = false,
  loadingMessage = "🚅 Loading logs...",
  noDataMessage = "No logs found",
  enableSorting = false,
}: DataTableProps<TData, TValue>) {
  const supportsExpansion = !!(renderSubComponent || renderChildRows) && !!getRowCanExpand;
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
    getRowId: (row: TData, index: number) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const _row: any = row as any;
      return _row?.request_id ?? String(index);
    },
    getCoreRowModel: getCoreRowModel(),
    ...(enableSorting && { getSortedRowModel: getSortedRowModel() }),
    ...(supportsExpansion && { getExpandedRowModel: getExpandedRowModel() }),
  });

  return (
    <div className="rounded-lg custom-border overflow-x-auto w-full max-w-full box-border">
      <Table className="[&_td]:py-0.5 [&_th]:py-1 table-fixed w-full box-border" style={{ minWidth: "400px" }}>
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
                    onClick={
                      canSort
                        ? header.column.getToggleSortingHandler()
                        : undefined
                    }
                  >
                    {header.isPlaceholder ? null : (
                      <div className="flex items-center gap-1">
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                        {canSort && (
                          <span className="text-muted-foreground">
                            {isSorted === "asc"
                              ? "↑"
                              : isSorted === "desc"
                                ? "↓"
                                : "⇅"}
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
                  className={`h-8 ${onRowClick ? "cursor-pointer hover:bg-muted" : ""}`}
                  onClick={() => onRowClick?.(row.original)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>

                {/* Child rows rendered as real table rows (MCP children) */}
                {supportsExpansion && row.getIsExpanded() && renderChildRows && (
                  renderChildRows({ row })
                )}

                {/* Legacy sub-component in colspan cell (audit logs) */}
                {supportsExpansion && row.getIsExpanded() && renderSubComponent && !renderChildRows && (
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
