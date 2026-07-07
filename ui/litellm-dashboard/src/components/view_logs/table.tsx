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
import { Skeleton } from "@/components/ui/skeleton";

declare module "@tanstack/react-table" {
  interface ColumnMeta<TData extends RowData, TValue> {
    numeric?: boolean;
    skeleton?: SkeletonMeta;
  }
}

type SkeletonMeta = {
  variant?: "text" | "pill" | "number" | "avatar";
  width?: string;
};

export const SKELETON_ROW_COUNT = 8;

function SkeletonCell({ numeric, skeleton }: { numeric?: boolean; skeleton?: SkeletonMeta }) {
  const variant = skeleton?.variant ?? (numeric ? "number" : "text");
  const width = skeleton?.width;

  if (variant === "pill") {
    return <Skeleton className={`h-5 rounded-full ${width ?? "w-16"}`} />;
  }
  if (variant === "number") {
    return <Skeleton className={`h-4 ml-auto ${width ?? "w-12"}`} />;
  }
  if (variant === "avatar") {
    return (
      <div className="flex items-center gap-2">
        <Skeleton className="h-4 w-4 rounded-full" />
        <Skeleton className={`h-4 ${width ?? "w-24"}`} />
      </div>
    );
  }
  return <Skeleton className={`h-4 ${width ?? "w-2/3"}`} />;
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
  /** Stale rows stay visible with a subtle fade while fresh data loads (no skeleton wipe) */
  isRefetching?: boolean;
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
  isRefetching = false,
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
        <TableBody className={isRefetching && !isLoading ? "opacity-60 transition-opacity" : ""}>
          {isLoading ? (
            Array.from({ length: SKELETON_ROW_COUNT }).map((_, rowIndex) => (
              <TableRow key={`skeleton-${rowIndex}`} className="h-8 hover:bg-transparent">
                {table.getVisibleLeafColumns().map((column) => (
                  <TableCell
                    key={column.id}
                    className="py-0.5 first:pl-4 last:pr-4"
                    style={hasExplicitColumnSizes ? { width: column.getSize() } : undefined}
                  >
                    <SkeletonCell numeric={column.columnDef.meta?.numeric} skeleton={column.columnDef.meta?.skeleton} />
                  </TableCell>
                ))}
              </TableRow>
            ))
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
