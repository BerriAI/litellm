"use client";

import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import {
  functionalUpdate,
  type ColumnFiltersState,
  type ColumnSort,
  type OnChangeFn,
  type PaginationState,
  type SortingState,
} from "@tanstack/react-table";
import { useCallback, useMemo } from "react";

import { useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import type { components } from "@/lib/http/schema";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

export type ResourceListQuery = Readonly<Record<string, string | number>>;

/** The management list envelope. The generated response models are monomorphic, so only `data` is generic here. */
export type ResourceListMeta = components["schemas"]["ListMeta"];

export interface ResourceListPage<TRow> {
  data: TRow[];
  meta: ResourceListMeta;
}

type FilterCodec = (filters: ColumnFiltersState) => ColumnFiltersState;

export interface ResourceListUrlState {
  sortFields: readonly string[];
  filterColumns: readonly string[];
  arrayFilterColumns?: readonly string[];
  keyPrefix?: string;
  urlKeys?: UrlTableStateOptions<string>["urlKeys"];
  toUrlFilters?: FilterCodec;
  fromUrlFilters?: FilterCodec;
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
  urlState: ResourceListUrlState;
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

const ARRAY_SEPARATOR = ",";
const NO_COLUMNS: readonly string[] = [];
const UNSORTED: ColumnSort = { id: "", desc: false };
const unchanged: FilterCodec = (filters) => filters;

const asStrings = (values: unknown[]): string[] => values.filter((value): value is string => typeof value === "string");

export const splitArrayFilters = (filters: ColumnFiltersState, arrayColumns: readonly string[]): ColumnFiltersState =>
  filters.map((filter) =>
    arrayColumns.includes(filter.id) && typeof filter.value === "string"
      ? {
          id: filter.id,
          value: filter.value
            .split(ARRAY_SEPARATOR)
            .map((entry) => entry.trim())
            .filter((entry) => entry !== ""),
        }
      : filter,
  );

export const joinArrayFilters = (filters: ColumnFiltersState, arrayColumns: readonly string[]): ColumnFiltersState =>
  filters.map((filter) =>
    arrayColumns.includes(filter.id) && Array.isArray(filter.value)
      ? { id: filter.id, value: asStrings(filter.value).join(ARRAY_SEPARATOR) }
      : filter,
  );

/**
 * State container for a table whose sorting, paging, search and filtering all run
 * on the server. Those four pieces of state live in the URL, get folded into one
 * JSON:API query, and come back as the exact props DataTable's server modes want.
 *
 * Empty parameters are dropped rather than sent blank because the management
 * routes reject query params they do not declare.
 */
export function useResourceList<TRow>(options: UseResourceListOptions<TRow>): ResourceListResult<TRow> {
  const { queryKey, fetchPage, serializeFilters, defaultSorting, defaultPageSize, enabled, urlState } = options;
  const arrayColumns = urlState.arrayFilterColumns ?? NO_COLUMNS;
  const toUrlFilters = urlState.toUrlFilters ?? unchanged;
  const fromUrlFilters = urlState.fromUrlFilters ?? unchanged;
  const { id: defaultSortId, desc: defaultSortDesc } = defaultSorting.at(0) ?? UNSORTED;

  const tableOptions = useMemo<UrlTableStateOptions<string>>(
    () => ({
      sortFields: urlState.sortFields,
      defaultSort: { id: defaultSortId, desc: defaultSortDesc },
      defaultPageSize,
      filterColumns: urlState.filterColumns,
      keyPrefix: urlState.keyPrefix,
      urlKeys: { search: "q", ...urlState.urlKeys },
    }),
    [urlState, defaultSortId, defaultSortDesc, defaultPageSize],
  );
  const table = useUrlTableState(tableOptions);
  const { pagination, onPaginationChange, search: searchValue, setSearch: onSearchChange } = table;
  const { sorting: urlSorting, onSortingChange: setUrlSorting } = table;
  const { columnFilters: urlFilters, onColumnFiltersChange: setUrlFilters } = table;

  const sorting = useMemo(() => urlSorting.filter((entry) => entry.id !== UNSORTED.id), [urlSorting]);
  const columnFilters = useMemo(
    () => fromUrlFilters(splitArrayFilters(urlFilters, arrayColumns)),
    [fromUrlFilters, urlFilters, arrayColumns],
  );
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
  const { data, isPending, isPlaceholderData, isFetching, error, refetch: refetchQuery } = useQuery(queryOptions);

  const onSortingChange = useCallback<OnChangeFn<SortingState>>(
    (updater) => setUrlSorting(functionalUpdate(updater, sorting)),
    [setUrlSorting, sorting],
  );

  const onColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>(
    (updater) => setUrlFilters(joinArrayFilters(toUrlFilters(functionalUpdate(updater, columnFilters)), arrayColumns)),
    [setUrlFilters, toUrlFilters, columnFilters, arrayColumns],
  );

  const refetch = useCallback(() => {
    void refetchQuery();
  }, [refetchQuery]);

  const rows = useMemo(() => data?.data ?? [], [data]);

  return {
    rows,
    rowCount: data?.meta.total_count ?? 0,
    isLoading: isPending || isPlaceholderData,
    isFetching,
    error,
    refetch,
    sorting,
    onSortingChange,
    pagination,
    onPaginationChange,
    columnFilters,
    onColumnFiltersChange,
    searchValue,
    onSearchChange,
  };
}
