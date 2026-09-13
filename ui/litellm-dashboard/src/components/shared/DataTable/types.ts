import type {
  ColumnDef,
  ColumnFiltersState,
  ExpandedState,
  OnChangeFn,
  PaginationState,
  Row,
  RowData,
  RowSelectionState,
  SortingState,
  Table,
  VisibilityState,
} from "@tanstack/react-table";
import type * as React from "react";

export type SortingMode = "none" | "client" | "server";
export type PaginationMode = "none" | "client" | "server";
export type FilterMode = "none" | "client" | "server";
export type ColumnResizeMode = "onEnd" | "onChange";
export type DataTableSize = "compact" | "default";
export type ColumnPinnedSide = "left" | "right";
export type DataTableSkeletonShape = "text" | "twoLine" | "badge" | "chips" | "meter";

export interface DataTableResolvedProps<TData extends RowData, TValue> {
  data: TData[];
  columns: ColumnDef<TData, TValue>[];
  getRowId?: (row: TData, index: number, parent?: Row<TData>) => string;

  isLoading?: boolean;
  loadingMessage?: string;
  skeletonRowCount?: number;
  noDataMessage?: React.ReactNode;

  sortingMode?: SortingMode;
  sorting?: SortingState;
  onSortingChange?: OnChangeFn<SortingState>;
  defaultSorting?: SortingState;
  enableSortingRemoval?: boolean;

  paginationMode?: PaginationMode;
  pagination?: PaginationState;
  onPaginationChange?: OnChangeFn<PaginationState>;
  rowCount?: number;
  pageSizeOptions?: number[];

  filterMode?: FilterMode;
  columnFilters?: ColumnFiltersState;
  onColumnFiltersChange?: OnChangeFn<ColumnFiltersState>;
  defaultColumnFilters?: ColumnFiltersState;

  globalFilter?: string;
  onGlobalFilterChange?: OnChangeFn<string>;

  enableColumnResizing?: boolean;
  columnResizeMode?: ColumnResizeMode;
  defaultColumnVisibility?: VisibilityState;

  getRowCanExpand?: (row: Row<TData>) => boolean;
  renderSubComponent?: (props: { row: Row<TData> }) => React.ReactElement;
  expanded?: ExpandedState;
  onExpandedChange?: OnChangeFn<ExpandedState>;

  enableRowSelection?: boolean | ((row: Row<TData>) => boolean);
  rowSelection?: RowSelectionState;
  onRowSelectionChange?: OnChangeFn<RowSelectionState>;

  onRowClick?: (row: TData) => void;

  rowClassName?: (row: Row<TData>) => string;

  maxBodyHeight?: number | string;
  /**
   * Scroll the rows inside whatever height the parent gives the table, rather than growing the page.
   * The table becomes a flex column, so the parent must be a height-constrained flex container; without
   * one it degrades to the normal auto-height layout. Use instead of `maxBodyHeight` to avoid a magic number.
   */
  fillHeight?: boolean;
  size?: DataTableSize;

  toolbar?: (table: Table<TData>) => React.ReactNode;
  paginationSlot?: (table: Table<TData>) => React.ReactNode;
  footer?: (table: Table<TData>) => React.ReactNode;
}

type DataTableBaseProps<TData extends RowData, TValue> = Omit<
  DataTableResolvedProps<TData, TValue>,
  | "sortingMode"
  | "sorting"
  | "onSortingChange"
  | "defaultSorting"
  | "paginationMode"
  | "pagination"
  | "onPaginationChange"
  | "rowCount"
  | "filterMode"
  | "columnFilters"
  | "onColumnFiltersChange"
  | "defaultColumnFilters"
  | "rowSelection"
  | "onRowSelectionChange"
>;

type SortingProps =
  | {
      sorting: SortingState;
      onSortingChange: OnChangeFn<SortingState>;
      sortingMode?: SortingMode;
      defaultSorting?: never;
    }
  | {
      sortingMode?: Exclude<SortingMode, "server">;
      sorting?: never;
      onSortingChange?: never;
      defaultSorting?: SortingState;
    };

type PaginationProps =
  | {
      paginationMode: "server";
      pagination: PaginationState;
      onPaginationChange: OnChangeFn<PaginationState>;
      rowCount: number;
    }
  | {
      paginationMode?: Exclude<PaginationMode, "server">;
      pagination?: PaginationState;
      onPaginationChange?: OnChangeFn<PaginationState>;
      rowCount?: number;
    };

type FilterProps =
  | {
      columnFilters: ColumnFiltersState;
      onColumnFiltersChange: OnChangeFn<ColumnFiltersState>;
      filterMode?: FilterMode;
      defaultColumnFilters?: never;
    }
  | {
      filterMode?: Exclude<FilterMode, "server">;
      columnFilters?: never;
      onColumnFiltersChange?: never;
      defaultColumnFilters?: ColumnFiltersState;
    };

type RowSelectionProps =
  | { rowSelection: RowSelectionState; onRowSelectionChange: OnChangeFn<RowSelectionState> }
  | { rowSelection?: never; onRowSelectionChange?: OnChangeFn<RowSelectionState> };

export type DataTableProps<TData extends RowData, TValue> = DataTableBaseProps<TData, TValue> &
  SortingProps &
  PaginationProps &
  FilterProps &
  RowSelectionProps;
