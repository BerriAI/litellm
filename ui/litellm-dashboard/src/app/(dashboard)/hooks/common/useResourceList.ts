"use client";

import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import type { ColumnFiltersState, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { useCallback, useMemo, useState } from "react";

import type { components } from "@/lib/http/schema";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

export type ResourceListQuery = Readonly<Record<string, string | number>>;

/** The management list envelope. The generated response models are monomorphic, so only `data` is generic here. */
export type ResourceListMeta = components["schemas"]["ListMeta"];

export interface ResourceListPage<TRow> {
  data: TRow[];
  meta: ResourceListMeta;
}

export interface UseResourceListOptions<TRow> {
  /** Prefix every list variant hangs off, so invalidating the resource root refetches whichever page is on screen. */
  queryKey: readonly unknown[];
  fetchPage: (query: ResourceListQuery, signal: AbortSignal) => Promise<ResourceListPage<TRow>>;
  /** Must be referentially stable; it feeds the query key. */
  serializeFilters: (filters: ColumnFiltersState) => Readonly<Record<string, string>>;
  defaultSorting: SortingState;
  defaultPageSize: number;
  enabled: boolean;
}

export interface ResourceListResult<TRow> {
  rows: TRow[];
  rowCount: number;
  isLoading: boolean;
  isFetching: boolean;
  error: Error | null;
  refetch: () => void;

  sorting: SortingState;
  onSortingChange: OnChangeFn<SortingState>;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  columnFilters: ColumnFiltersState;
  onColumnFiltersChange: OnChangeFn<ColumnFiltersState>;
  searchValue: string;
  onSearchChange: (value: string) => void;
}

/** JSON:API sort form: comma separated fields, `-` prefix for descending. */
export const toSortParam = (sorting: SortingState): string =>
  sorting.map((entry) => (entry.desc ? `-${entry.id}` : entry.id)).join(",");

/**
 * State container for a table whose sorting, paging, search and filtering all run
 * on the server. It owns those four pieces of state, folds them into one JSON:API
 * query, and returns the exact props DataTable's server modes want.
 *
 * Empty parameters are dropped rather than sent blank because the management
 * routes reject query params they do not declare.
 */
export function useResourceList<TRow>(options: UseResourceListOptions<TRow>): ResourceListResult<TRow> {
  const { queryKey, fetchPage, serializeFilters, defaultSorting, defaultPageSize, enabled } = options;

  const [sorting, setSorting] = useState<SortingState>(defaultSorting);
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: defaultPageSize });
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [searchValue, setSearchValue] = useState("");
  const [debouncedSearch] = useDebouncedValue(searchValue, { wait: DEBOUNCE_WAIT_MS });

  const query = useMemo<ResourceListQuery>(() => {
    const sort = toSortParam(sorting);
    const search = debouncedSearch.trim();
    return {
      page: pagination.pageIndex + 1,
      page_size: pagination.pageSize,
      ...(sort === "" ? {} : { sort }),
      ...(search === "" ? {} : { q: search }),
      ...serializeFilters(columnFilters),
    };
  }, [sorting, pagination.pageIndex, pagination.pageSize, debouncedSearch, columnFilters, serializeFilters]);

  const queryOptions: UseQueryOptions<ResourceListPage<TRow>, Error, ResourceListPage<TRow>, readonly unknown[]> = {
    queryKey: [...queryKey, query],
    queryFn: ({ signal }) => fetchPage(query, signal),
    enabled,
    placeholderData: (previous) => previous,
  };
  const { data, isLoading, isFetching, error, refetch: refetchQuery } = useQuery(queryOptions);

  const toFirstPage = useCallback(() => setPagination((previous) => ({ ...previous, pageIndex: 0 })), []);

  const onSortingChange = useCallback<OnChangeFn<SortingState>>(
    (updater) => {
      setSorting(updater);
      toFirstPage();
    },
    [toFirstPage],
  );

  const onColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>(
    (updater) => {
      setColumnFilters(updater);
      toFirstPage();
    },
    [toFirstPage],
  );

  const onSearchChange = useCallback(
    (value: string) => {
      setSearchValue(value);
      toFirstPage();
    },
    [toFirstPage],
  );

  const refetch = useCallback(() => {
    void refetchQuery();
  }, [refetchQuery]);

  const rows = useMemo(() => data?.data ?? [], [data]);

  return {
    rows,
    rowCount: data?.meta.total_count ?? 0,
    isLoading,
    isFetching,
    error,
    refetch,
    sorting,
    onSortingChange,
    pagination,
    onPaginationChange: setPagination,
    columnFilters,
    onColumnFiltersChange,
    searchValue,
    onSearchChange,
  };
}
