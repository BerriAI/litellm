"use client";

import {
  type Cell,
  type Column,
  type ColumnDef,
  type ColumnFiltersState,
  type ColumnPinningState,
  type ColumnSizingState,
  type ExpandedState,
  flexRender,
  getCoreRowModel,
  getExpandedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  type Header,
  type OnChangeFn,
  type PaginationState,
  type Row,
  type RowData,
  type RowSelectionState,
  type Table,
  type TableOptions,
  useReactTable,
  type VisibilityState,
} from "@tanstack/react-table";
import { SearchX } from "lucide-react";
import * as React from "react";
import { Fragment, useEffect, useState } from "react";

import { Skeleton } from "@/components/ui/skeleton";
import {
  NUMERIC_CELL_CLASS,
  Table as TableRoot,
  TableBody,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn, cva } from "@/lib/cva.config";

import "./columnMeta";
import { DataTablePagination, DEFAULT_PAGE_SIZE_OPTIONS } from "./DataTablePagination";
import type {
  ColumnPinnedSide,
  DataTableProps,
  DataTableResolvedProps,
  DataTableSize,
  FilterMode,
  PaginationMode,
  SortingMode,
} from "./types";

const INTERACTIVE_SELECTOR = "button, a, input, select, textarea, [role=checkbox], [data-row-click-exempt]";

const noop = () => {};

const dataTableRoot = cva("w-full", {
  variants: { fill: { true: "flex h-full min-h-0 flex-1 flex-col", false: null } },
});

const dataTableFrame = cva("overflow-hidden rounded-lg border border-border", {
  variants: { fill: { true: "flex min-h-0 flex-1 flex-col", false: null } },
});

const dataTableScroller = cva("", {
  variants: {
    sticky: { true: "overflow-auto [&_[data-slot=table-container]]:overflow-visible", false: "overflow-x-auto" },
    fill: { true: "min-h-0 flex-1", false: null },
    stretchEmpty: { true: "[container-type:inline-size] [&_[data-slot=table-container]]:h-full", false: null },
  },
});

const dataTableTable = cva("", {
  variants: {
    resizable: { true: "table-fixed", false: null },
    stretchEmpty: { true: "h-full", false: null },
  },
});

const dataTableHeader = cva("", {
  variants: { sticky: { true: "sticky top-0 z-sticky bg-background", false: null } },
});

const dataTableBody = cva("", {
  variants: { stretchEmpty: { true: "h-full", false: null } },
});

const messageCell = cva("h-24 text-center align-middle text-sm whitespace-normal text-muted-foreground", {
  variants: { stretch: { true: "p-0", false: null } },
});

function columnDefId<TData, TValue>(column: ColumnDef<TData, TValue>): string | undefined {
  if ("id" in column && typeof column.id === "string") {
    return column.id;
  }
  if ("accessorKey" in column && column.accessorKey != null) {
    return String(column.accessorKey);
  }
  return undefined;
}

function derivePinning<TData, TValue>(columns: ColumnDef<TData, TValue>[]): ColumnPinningState {
  const collect = (side: ColumnPinnedSide): string[] =>
    columns
      .filter((column) => column.meta?.pinned === side)
      .map(columnDefId)
      .filter((id): id is string => id !== undefined);
  return { left: collect("left"), right: collect("right") };
}

function columnCanGlobalFilter<TData>(firstRow: TData | undefined, column: Column<TData, unknown>): boolean {
  if (column.columnDef.enableGlobalFilter === true) return true;
  if (firstRow === undefined || column.accessorFn === undefined) return false;
  const firstValue: unknown = column.accessorFn(firstRow, 0);
  return typeof firstValue === "string" || typeof firstValue === "number";
}

function buildRowModels<TData>(
  sortingMode: SortingMode,
  paginationMode: PaginationMode,
  filterMode: FilterMode,
  getRowCanExpand: ((row: Row<TData>) => boolean) | undefined,
): Partial<TableOptions<TData>> {
  return {
    ...(filterMode === "client" ? { getFilteredRowModel: getFilteredRowModel() } : {}),
    ...(sortingMode === "client" ? { getSortedRowModel: getSortedRowModel() } : {}),
    ...(paginationMode === "client" ? { getPaginationRowModel: getPaginationRowModel() } : {}),
    ...(getRowCanExpand !== undefined ? { getRowCanExpand, getExpandedRowModel: getExpandedRowModel() } : {}),
  };
}

function stickyLayer(isPinned: boolean, isHeader: boolean): string {
  if (isPinned && isHeader) {
    return "z-sticky-pinned";
  }
  if (isHeader) {
    return "z-sticky";
  }
  return "z-raised";
}

function pinnedShadow(pinned: false | ColumnPinnedSide): string {
  if (pinned === "left") {
    return "shadow-[inset_-1px_0_0_var(--color-border)]";
  }
  if (pinned === "right") {
    return "shadow-[inset_1px_0_0_var(--color-border)]";
  }
  return "";
}

function computeStickyStyle<TData, TValue>(
  column: Column<TData, TValue>,
  isHeader: boolean,
  stickyHeader: boolean,
): { style: React.CSSProperties; className: string } {
  const pinned = column.getIsPinned();
  const stickyTop = isHeader && stickyHeader;
  if (!pinned && !stickyTop) {
    return { style: {}, className: "" };
  }

  const left = pinned === "left" ? column.getStart("left") : undefined;
  const right = pinned === "right" ? column.getAfter("right") : undefined;

  const style: React.CSSProperties = {
    position: "sticky",
    ...(stickyTop ? { top: 0 } : {}),
    ...(left !== undefined ? { left } : {}),
    ...(right !== undefined ? { right } : {}),
  };

  return {
    style,
    className: cn(stickyLayer(pinned !== false, isHeader), pinned ? "bg-background" : "", pinnedShadow(pinned)),
  };
}

function widthStyle<TData, TValue>(
  column: Column<TData, TValue>,
  enableColumnResizing: boolean,
): React.CSSProperties | undefined {
  if (enableColumnResizing || column.columnDef.size !== undefined) {
    return { width: column.getSize() };
  }
  return undefined;
}

interface HeadCellProps<TData> {
  header: Header<TData, unknown>;
  size: DataTableSize;
  stickyHeader: boolean;
  enableColumnResizing: boolean;
}

function DataTableHeadCell<TData>({ header, size, stickyHeader, enableColumnResizing }: HeadCellProps<TData>) {
  const { column } = header;
  const meta = column.columnDef.meta;
  const sticky = computeStickyStyle(column, true, stickyHeader);
  const canResize = enableColumnResizing && column.getCanResize();

  return (
    <TableHead
      data-header-id={header.id}
      className={cn(
        "relative text-muted-foreground",
        size === "compact" ? "h-8 px-2 py-1 text-xs" : "",
        meta?.numeric ? NUMERIC_CELL_CLASS : "",
        meta?.className,
        meta?.headerClassName,
        sticky.className,
      )}
      style={{ ...sticky.style, ...widthStyle(column, enableColumnResizing) }}
    >
      {header.isPlaceholder ? null : (
        <div className={cn("flex items-center gap-1", meta?.numeric ? "justify-end" : "")}>
          {flexRender(column.columnDef.header, header.getContext())}
        </div>
      )}
      {canResize && (
        <div
          data-testid={`column-resizer-${header.id}`}
          onMouseDown={header.getResizeHandler()}
          onTouchStart={header.getResizeHandler()}
          onDoubleClick={() => column.resetSize()}
          className={cn(
            "absolute top-0 right-0 h-full w-1 cursor-col-resize touch-none select-none hover:bg-border",
            column.getIsResizing() ? "bg-primary" : "",
          )}
        />
      )}
    </TableHead>
  );
}

interface BodyCellProps<TData> {
  cell: Cell<TData, unknown>;
  size: DataTableSize;
  stickyHeader: boolean;
  enableColumnResizing: boolean;
}

function DataTableBodyCell<TData>({ cell, size, stickyHeader, enableColumnResizing }: BodyCellProps<TData>) {
  const { column } = cell;
  const meta = column.columnDef.meta;
  const sticky = computeStickyStyle(column, false, stickyHeader);

  return (
    <TableCell
      className={cn(
        "overflow-hidden text-ellipsis",
        size === "compact" ? "px-2 py-1 text-xs" : "",
        meta?.numeric ? NUMERIC_CELL_CLASS : "",
        meta?.className,
        sticky.className,
      )}
      style={{ ...sticky.style, ...widthStyle(column, enableColumnResizing) }}
    >
      {flexRender(column.columnDef.cell, cell.getContext())}
    </TableCell>
  );
}

interface BodyRowProps<TData> {
  row: Row<TData>;
  size: DataTableSize;
  stickyHeader: boolean;
  enableColumnResizing: boolean;
  onRowClick?: (row: TData) => void;
  rowClassName?: (row: Row<TData>) => string;
  renderSubComponent?: (props: { row: Row<TData> }) => React.ReactElement;
}

function DataTableBodyRow<TData>({
  row,
  size,
  stickyHeader,
  enableColumnResizing,
  onRowClick,
  rowClassName,
  renderSubComponent,
}: BodyRowProps<TData>) {
  const clickable = onRowClick !== undefined;
  const cells = row.getVisibleCells();

  const handleClick = (event: React.MouseEvent<HTMLTableRowElement>) => {
    if (onRowClick === undefined) {
      return;
    }
    const target = event.target as HTMLElement | null;
    if (target === null || !event.currentTarget.contains(target)) {
      return;
    }
    if (target.closest(INTERACTIVE_SELECTOR) !== null) {
      return;
    }
    onRowClick(row.original);
  };

  return (
    <Fragment>
      <TableRow
        data-row-id={row.id}
        className={cn(clickable ? "cursor-pointer" : "", size === "compact" ? "h-8" : "", rowClassName?.(row))}
        onClick={clickable ? handleClick : undefined}
      >
        {cells.map((cell) => (
          <DataTableBodyCell
            key={cell.id}
            cell={cell}
            size={size}
            stickyHeader={stickyHeader}
            enableColumnResizing={enableColumnResizing}
          />
        ))}
      </TableRow>
      {renderSubComponent !== undefined && row.getIsExpanded() && (
        <TableRow className="hover:bg-transparent">
          <TableCell colSpan={cells.length} className="p-0">
            {renderSubComponent({ row })}
          </TableCell>
        </TableRow>
      )}
    </Fragment>
  );
}

function MessageRow({
  colSpan,
  children,
  stretch = false,
}: {
  colSpan: number;
  children: React.ReactNode;
  stretch?: boolean;
}) {
  return (
    <TableRow className="hover:bg-transparent">
      <TableCell colSpan={colSpan} className={messageCell({ stretch })}>
        {stretch ? (
          <div className="sticky left-0 flex h-full w-[100cqw] items-center justify-center">{children}</div>
        ) : (
          children
        )}
      </TableCell>
    </TableRow>
  );
}

function DefaultEmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <SearchX className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No results</div>
      <div className="text-sm text-muted-foreground">No rows match your search or filters.</div>
    </div>
  );
}

const SKELETON_WIDTHS = ["w-[58%]", "w-[44%]", "w-[70%]", "w-[50%]", "w-[64%]", "w-[48%]"] as const;

function SkeletonCell<TData>({ column, index }: { column: Column<TData, unknown> | undefined; index: number }) {
  const meta = column?.columnDef.meta;
  const width = SKELETON_WIDTHS[index % SKELETON_WIDTHS.length];
  const shape = meta?.skeleton;
  if (meta?.renderSkeleton !== undefined) {
    return <>{meta.renderSkeleton()}</>;
  }
  if (shape === "twoLine") {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className={cn("h-3.5", width)} />
        <Skeleton className="h-2.5 w-2/5 opacity-65" />
      </div>
    );
  }
  if (shape === "badge") {
    return <Skeleton className={cn("h-5 w-16 rounded-full", meta?.numeric ? "ml-auto" : "")} />;
  }
  if (shape === "chips") {
    return (
      <div className="flex items-center gap-1.5">
        <Skeleton className="h-5 w-14 rounded-full" />
        <Skeleton className="h-5 w-20 rounded-full" />
        <Skeleton className="h-5 w-9 rounded-full opacity-65" />
      </div>
    );
  }
  if (shape === "meter") {
    return (
      <div className="flex flex-col gap-1.5">
        <Skeleton className="h-3.5 w-24" />
        <Skeleton className="h-1.5 w-full rounded-full" />
      </div>
    );
  }
  return <Skeleton className={cn("h-3.5", width, meta?.numeric ? "ml-auto" : "")} />;
}

function SkeletonRows<TData>({
  rowCount,
  columns,
  size,
  message,
}: {
  rowCount: number;
  columns: readonly Column<TData, unknown>[];
  size: DataTableSize;
  message?: string;
}) {
  const rowKeys = Array.from({ length: Math.max(rowCount, 1) }, (_, index) => index);
  const cells = columns.length > 0 ? columns : [undefined];
  return (
    <Fragment>
      {rowKeys.map((rowKey) => (
        <TableRow
          key={`skeleton-${rowKey}`}
          className={cn("hover:bg-transparent", size === "compact" ? "h-8" : "")}
          data-testid="skeleton-row"
        >
          {cells.map((column, columnKey) => (
            <TableCell key={column?.id ?? columnKey} className={size === "compact" ? "px-2 py-1" : ""}>
              <SkeletonCell column={column} index={columnKey} />
              {rowKey === 0 && columnKey === 0 && message !== undefined ? (
                <span className="sr-only">{message}</span>
              ) : null}
            </TableCell>
          ))}
        </TableRow>
      ))}
    </Fragment>
  );
}

function useControllable<T>(
  controlled: T | undefined,
  controlledOnChange: OnChangeFn<T> | undefined,
  initial: T,
): { value: T; onChange: OnChangeFn<T> } {
  const [internal, setInternal] = useState<T>(initial);
  if (controlled !== undefined) {
    return { value: controlled, onChange: controlledOnChange ?? noop };
  }
  return { value: internal, onChange: setInternal };
}

function usePageClamp(
  active: boolean,
  rowCount: number | undefined,
  pagination: { value: PaginationState; onChange: OnChangeFn<PaginationState> },
): void {
  const { pageIndex, pageSize } = pagination.value;
  const { onChange } = pagination;
  useEffect(() => {
    if (!active || rowCount === undefined) return;
    const lastPageIndex = Math.max(Math.ceil(rowCount / pageSize) - 1, 0);
    if (pageIndex <= lastPageIndex) return;
    onChange({ pageIndex: lastPageIndex, pageSize });
  }, [active, rowCount, pageIndex, pageSize, onChange]);
}

function useDataTableInstance<TData extends RowData, TValue>(
  props: DataTableResolvedProps<TData, TValue>,
): Table<TData> {
  const {
    data,
    columns,
    getRowId,
    sortingMode = "none",
    sorting,
    onSortingChange,
    defaultSorting,
    enableSortingRemoval = false,
    paginationMode = "none",
    pagination,
    onPaginationChange,
    rowCount,
    isLoading = false,
    isError,
    pageSizeOptions = DEFAULT_PAGE_SIZE_OPTIONS,
    filterMode = "none",
    columnFilters,
    onColumnFiltersChange,
    defaultColumnFilters,
    globalFilter,
    onGlobalFilterChange,
    enableColumnResizing = false,
    columnResizeMode = "onEnd",
    columnVisibility,
    onColumnVisibilityChange,
    defaultColumnVisibility,
    getRowCanExpand,
    renderSubComponent,
    expanded,
    onExpandedChange,
    enableRowSelection,
    rowSelection,
    onRowSelectionChange,
  } = props;

  const sortingState = useControllable(sorting, onSortingChange, defaultSorting ?? []);
  const paginationState = useControllable(pagination, onPaginationChange, {
    pageIndex: 0,
    pageSize: pageSizeOptions[0] ?? 25,
  });
  const filterState = useControllable<ColumnFiltersState>(
    columnFilters,
    onColumnFiltersChange,
    defaultColumnFilters ?? [],
  );
  const globalFilterState = useControllable<string>(globalFilter, onGlobalFilterChange, "");
  const expandedState = useControllable<ExpandedState>(expanded, onExpandedChange, {});
  const rowSelectionState = useControllable<RowSelectionState>(rowSelection, onRowSelectionChange, {});
  const columnVisibilityState = useControllable<VisibilityState>(
    columnVisibility,
    onColumnVisibilityChange,
    defaultColumnVisibility ?? {},
  );
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>({});
  const columnPinning = React.useMemo(() => derivePinning(columns), [columns]);
  const expansionGuard = renderSubComponent !== undefined ? getRowCanExpand : undefined;

  const tableOptions: TableOptions<TData> = {
    data,
    columns,
    state: {
      sorting: sortingState.value,
      pagination: paginationState.value,
      columnFilters: filterState.value,
      globalFilter: globalFilterState.value,
      expanded: expandedState.value,
      rowSelection: rowSelectionState.value,
      columnVisibility: columnVisibilityState.value,
      columnSizing,
    },
    initialState: { columnPinning },
    manualSorting: sortingMode === "server",
    manualPagination: paginationMode === "server",
    manualFiltering: filterMode === "server",
    enableSortingRemoval,
    enableColumnResizing,
    columnResizeMode,
    onSortingChange: sortingState.onChange,
    onPaginationChange: paginationState.onChange,
    onColumnFiltersChange: filterState.onChange,
    onGlobalFilterChange: globalFilterState.onChange,
    onExpandedChange: expandedState.onChange,
    onRowSelectionChange: rowSelectionState.onChange,
    onColumnVisibilityChange: columnVisibilityState.onChange,
    onColumnSizingChange: setColumnSizing,
    getColumnCanGlobalFilter: (column) => columnCanGlobalFilter(data[0], column),
    getCoreRowModel: getCoreRowModel(),
    ...buildRowModels(sortingMode, paginationMode, filterMode, expansionGuard),
    ...(getRowId !== undefined ? { getRowId } : {}),
    ...(enableRowSelection !== undefined ? { enableRowSelection } : {}),
    ...(paginationMode === "server" && rowCount !== undefined ? { rowCount } : {}),
    autoResetPageIndex: pagination === undefined && paginationMode !== "server",
  };

  const table = useReactTable(tableOptions);
  const clampOptions: SettledPageClampOptions = {
    paginationMode,
    controlled: pagination !== undefined,
    settled: !isLoading && !isError,
    rowCount,
    pagination: paginationState,
  };
  useSettledPageClamp(table, clampOptions);
  return table;
}

type SettledPageClampOptions = {
  paginationMode: PaginationMode;
  controlled: boolean;
  settled: boolean;
  rowCount: number | undefined;
  pagination: { value: PaginationState; onChange: OnChangeFn<PaginationState> };
};

function useSettledPageClamp<TData extends RowData>(table: Table<TData>, options: SettledPageClampOptions): void {
  const { paginationMode, controlled, settled, rowCount, pagination } = options;
  const clientRowCount = paginationMode === "client" ? table.getPrePaginationRowModel().rows.length : 0;
  const clientPageIsClampable = paginationMode === "client" && controlled && clientRowCount > 0;
  usePageClamp(
    settled && (paginationMode === "server" || clientPageIsClampable),
    paginationMode === "server" ? rowCount : clientRowCount,
    pagination,
  );
}

export function DataTable<TData extends RowData, TValue>(props: DataTableProps<TData, TValue>) {
  const resolved: DataTableResolvedProps<TData, TValue> = props;

  const {
    isLoading = false,
    loadingMessage = "Loading…",
    skeletonRowCount = 8,
    noDataMessage,
    paginationMode = "none",
    rowCount,
    pageSizeOptions = DEFAULT_PAGE_SIZE_OPTIONS,
    enableColumnResizing = false,
    onRowClick,
    rowClassName,
    renderSubComponent,
    maxBodyHeight,
    fillHeight = false,
    size = "default",
    toolbar,
    paginationSlot,
    footer,
  } = resolved;

  const table = useDataTableInstance(resolved);

  const rows = table.getRowModel().rows;
  const visibleColumnCount = table.getVisibleLeafColumns().length;
  const stickyHeader = maxBodyHeight !== undefined || fillHeight;
  const stretchEmpty = fillHeight && !isLoading && rows.length === 0;
  const tableStyle = enableColumnResizing ? { width: table.getTotalSize(), minWidth: "100%" } : undefined;

  const renderPagination = (): React.ReactNode => {
    if (paginationSlot !== undefined) {
      return paginationSlot(table);
    }
    if (paginationMode === "none") {
      return null;
    }
    const current = table.getState().pagination;
    const total = paginationMode === "server" ? rowCount ?? 0 : table.getPrePaginationRowModel().rows.length;
    return (
      <DataTablePagination
        page={current.pageIndex}
        pageSize={current.pageSize}
        rowCount={total}
        onPageChange={(next) => table.setPageIndex(next)}
        onPageSizeChange={(next) => table.setPageSize(next)}
        pageSizeOptions={pageSizeOptions}
        isLoading={isLoading}
      />
    );
  };

  const renderBody = (): React.ReactNode => {
    if (isLoading) {
      return (
        <SkeletonRows
          rowCount={skeletonRowCount}
          columns={table.getVisibleLeafColumns()}
          size={size}
          message={loadingMessage}
        />
      );
    }
    if (rows.length === 0) {
      return (
        <MessageRow colSpan={visibleColumnCount} stretch={stretchEmpty}>
          {noDataMessage ?? <DefaultEmptyState />}
        </MessageRow>
      );
    }
    return rows.map((row) => (
      <DataTableBodyRow
        key={row.id}
        row={row}
        size={size}
        stickyHeader={stickyHeader}
        enableColumnResizing={enableColumnResizing}
        onRowClick={onRowClick}
        rowClassName={rowClassName}
        renderSubComponent={renderSubComponent}
      />
    ));
  };

  const paginationNode = renderPagination();

  return (
    <div data-testid="data-table-root" className={dataTableRoot({ fill: fillHeight })}>
      <div data-testid="data-table-frame" className={dataTableFrame({ fill: fillHeight })}>
        {toolbar !== undefined && <div className="shrink-0 border-b border-border px-4 py-3">{toolbar(table)}</div>}
        <div
          data-testid="data-table-scroller"
          className={dataTableScroller({ sticky: stickyHeader, fill: fillHeight, stretchEmpty })}
          style={maxBodyHeight !== undefined ? { maxHeight: maxBodyHeight } : undefined}
        >
          <TableRoot className={dataTableTable({ resizable: enableColumnResizing, stretchEmpty })} style={tableStyle}>
            <TableHeader data-testid="data-table-head" className={dataTableHeader({ sticky: stickyHeader })}>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id} className="bg-muted/50">
                  {headerGroup.headers.map((header) => (
                    <DataTableHeadCell
                      key={header.id}
                      header={header}
                      size={size}
                      stickyHeader={stickyHeader}
                      enableColumnResizing={enableColumnResizing}
                    />
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody className={dataTableBody({ stretchEmpty })}>{renderBody()}</TableBody>
            {footer !== undefined && <TableFooter>{footer(table)}</TableFooter>}
          </TableRoot>
        </div>
        {paginationNode !== null && <div className="shrink-0 border-t border-border">{paginationNode}</div>}
      </div>
    </div>
  );
}
