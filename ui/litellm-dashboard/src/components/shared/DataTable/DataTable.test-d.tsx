import type { ColumnDef, PaginationState, RowSelectionState, SortingState } from "@tanstack/react-table";

import { DataTable } from "./DataTable";

interface Row {
  id: string;
  name: string;
}

const data: Row[] = [];
const columns: ColumnDef<Row, unknown>[] = [];
const sorting: SortingState = [{ id: "name", desc: false }];
const pagination: PaginationState = { pageIndex: 0, pageSize: 10 };
const rowSelection: RowSelectionState = { r1: true };
const noop = () => {};

export const uncontrolled = <DataTable data={data} columns={columns} defaultSorting={sorting} />;

export const controlled = (
  <DataTable
    data={data}
    columns={columns}
    sortingMode="server"
    sorting={sorting}
    onSortingChange={noop}
    paginationMode="server"
    pagination={pagination}
    onPaginationChange={noop}
    rowCount={0}
    filterMode="server"
    columnFilters={[]}
    onColumnFiltersChange={noop}
    rowSelection={rowSelection}
    onRowSelectionChange={noop}
  />
);

export const clientSortingWithServerPagination = (
  <DataTable
    data={data}
    columns={columns}
    defaultSorting={sorting}
    paginationMode="server"
    pagination={pagination}
    onPaginationChange={noop}
    rowCount={0}
  />
);

// @ts-expect-error sortingMode="server" requires `sorting` and `onSortingChange`
export const serverSortingWithoutState = <DataTable data={data} columns={columns} sortingMode="server" />;

// @ts-expect-error paginationMode="server" requires `pagination`, `onPaginationChange` and `rowCount`
export const serverPaginationWithoutState = <DataTable data={data} columns={columns} paginationMode="server" />;

// @ts-expect-error filterMode="server" requires `columnFilters` and `onColumnFiltersChange`
export const serverFilteringWithoutState = <DataTable data={data} columns={columns} filterMode="server" />;

export const bothSortingSources = (
  // @ts-expect-error `defaultSorting` seeds uncontrolled sorting, so it cannot pair with a controlled `sorting`
  <DataTable data={data} columns={columns} defaultSorting={sorting} sorting={sorting} onSortingChange={noop} />
);

export const selectionWithoutHandler = (
  // @ts-expect-error a controlled `rowSelection` needs `onRowSelectionChange` or selection changes are dropped
  <DataTable data={data} columns={columns} rowSelection={rowSelection} />
);
