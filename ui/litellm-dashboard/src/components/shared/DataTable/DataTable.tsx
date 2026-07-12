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
  type Row,
  type RowData,
  type Table,
  type TableOptions,
  useReactTable,
  type VisibilityState,
} from "@tanstack/react-table";
import { SearchX } from "lucide-react";
import * as React from "react";
import { Fragment, useState } from "react";

import { Skeleton } from "@/components/ui/skeleton";
import {
  Table as TableRoot,
  TableBody,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/cva.config";

import "./columnMeta";
import { DataTablePagination, DEFAULT_PAGE_SIZE_OPTIONS } from "./DataTablePagination";
import type { ColumnPinnedSide, DataTableProps, DataTableSize, FilterMode, PaginationMode, SortingMode } from "./types";

const INTERACTIVE_SELECTOR = "button, a, input, select, textarea, [role=checkbox], [data-row-click-exempt]";

const noop = () => {};

export class DataTableConfigError extends Error {
  constructor(messages: readonly string[]) {
    super(`DataTable misconfiguration:\n- ${messages.join("\n- ")}`);
    this.name = "DataTableConfigError";
  }
}

export function validateDataTableConfig<TData extends RowData, TValue>(
  props: DataTableProps<TData, TValue>,
): readonly string[] {
  const serverSortingIncomplete =
    props.sortingMode === "server" && (props.sorting === undefined || props.onSortingChange === undefined);

  const serverPaginationPropsMissing =
    props.pagination === undefined || props.onPaginationChange === undefined || props.rowCount === undefined;
  const serverPaginationIncomplete = props.paginationMode === "server" && serverPaginationPropsMissing;

  const serverFilteringIncomplete =
    props.filterMode === "server" && (props.columnFilters === undefined || props.onColumnFiltersChange === undefined);

  const bothSortingSources = props.defaultSorting !== undefined && props.sorting !== undefined;
  const bothFilterSources = props.defaultColumnFilters !== undefined && props.columnFilters !== undefined;

  return [
    serverSortingIncomplete ? "sortingMode='server' requires both `sorting` and `onSortingChange`." : null,
    serverPaginationIncomplete
      ? "paginationMode='server' requires `pagination`, `onPaginationChange`, and `rowCount`."
      : null,
    serverFilteringIncomplete ? "filterMode='server' requires both `columnFilters` and `onColumnFiltersChange`." : null,
    bothSortingSources ? "Provide either `defaultSorting` (uncontrolled) or `sorting` (controlled), not both." : null,
    bothFilterSources
      ? "Provide either `defaultColumnFilters` (uncontrolled) or `columnFilters` (controlled), not both."
      : null,
  ].filter((message): message is string => message !== null);
}

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

function stickyZIndex(isPinned: boolean, isHeader: boolean): number {
  if (isPinned && isHeader) {
    return 30;
  }
  if (isHeader) {
    return 20;
  }
  return 10;
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
    zIndex: stickyZIndex(pinned !== false, isHeader),
    ...(stickyTop ? { top: 0 } : {}),
    ...(left !== undefined ? { left } : {}),
    ...(right !== undefined ? { right } : {}),
  };

  return { style, className: cn(pinned ? "bg-background" : "", pinnedShadow(pinned)) };
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
        meta?.numeric ? "text-right" : "",
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
          data-resizer
          data-header-id={header.id}
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
        meta?.numeric ? "text-right tabular-nums" : "",
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

function MessageRow({ colSpan, children }: { colSpan: number; children: React.ReactNode }) {
  return (
    <TableRow className="hover:bg-transparent">
      <TableCell colSpan={colSpan} className="h-24 text-center align-middle text-sm text-muted-foreground">
        {children}
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
  if (meta?.skeleton === "twoLine") {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className={cn("h-3.5", width)} />
        <Skeleton className="h-2.5 w-2/5 opacity-65" />
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
        <TableRow key={`skeleton-${rowKey}`} className="hover:bg-transparent" data-testid="skeleton-row">
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

function useDataTableInstance<TData extends RowData, TValue>(props: DataTableProps<TData, TValue>): Table<TData> {
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
    pageSizeOptions = DEFAULT_PAGE_SIZE_OPTIONS,
    filterMode = "none",
    columnFilters,
    onColumnFiltersChange,
    defaultColumnFilters,
    globalFilter,
    onGlobalFilterChange,
    enableColumnResizing = false,
    columnResizeMode = "onEnd",
    defaultColumnVisibility,
    getRowCanExpand,
    renderSubComponent,
    expanded,
    onExpandedChange,
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
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(defaultColumnVisibility ?? {});
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
      columnVisibility,
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
    onColumnVisibilityChange: setColumnVisibility,
    onColumnSizingChange: setColumnSizing,
    getCoreRowModel: getCoreRowModel(),
    ...buildRowModels(sortingMode, paginationMode, filterMode, expansionGuard),
    ...(getRowId !== undefined ? { getRowId } : {}),
    ...(paginationMode === "server" && rowCount !== undefined ? { rowCount } : {}),
  };

  return useReactTable(tableOptions);
}

export function DataTable<TData extends RowData, TValue>(props: DataTableProps<TData, TValue>) {
  // Validate once at construction so a misconfig surfaces immediately instead of on every render.
  useState<null>(() => {
    const errors = validateDataTableConfig(props);
    if (errors.length > 0) {
      throw new DataTableConfigError(errors);
    }
    return null;
  });

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
    size = "default",
    toolbar,
    paginationSlot,
    footer,
  } = props;

  const table = useDataTableInstance(props);

  const rows = table.getRowModel().rows;
  const visibleColumnCount = table.getVisibleLeafColumns().length;
  const stickyHeader = maxBodyHeight !== undefined;
  const tableStyle = enableColumnResizing ? { width: table.getTotalSize() } : undefined;

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
      return <MessageRow colSpan={visibleColumnCount}>{noDataMessage ?? <DefaultEmptyState />}</MessageRow>;
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
    <div className="w-full">
      <div className="overflow-hidden rounded-lg border border-border">
        {toolbar !== undefined && <div className="border-b border-border px-4 py-3">{toolbar(table)}</div>}
        <div
          className={stickyHeader ? "overflow-auto" : "overflow-x-auto"}
          style={stickyHeader ? { maxHeight: maxBodyHeight } : undefined}
        >
          <TableRoot className={enableColumnResizing ? "table-fixed" : ""} style={tableStyle}>
            <TableHeader className={stickyHeader ? "sticky top-0 z-20" : ""}>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id} className="bg-muted/50 hover:bg-muted/50">
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
            <TableBody>{renderBody()}</TableBody>
            {footer !== undefined && <TableFooter>{footer(table)}</TableFooter>}
          </TableRoot>
        </div>
        {paginationNode !== null && <div className="border-t border-border">{paginationNode}</div>}
      </div>
    </div>
  );
}
