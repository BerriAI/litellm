import type {
  ColumnDef,
  ColumnFiltersState,
  ExpandedState,
  OnChangeFn,
  PaginationState,
  Row,
  RowData,
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

export interface DataTableProps<TData extends RowData, TValue> {
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

  onRowClick?: (row: TData) => void;

  rowClassName?: (row: Row<TData>) => string;

  maxBodyHeight?: number | string;
  size?: DataTableSize;

  toolbar?: (table: Table<TData>) => React.ReactNode;
  paginationSlot?: (table: Table<TData>) => React.ReactNode;
  footer?: (table: Table<TData>) => React.ReactNode;
}
