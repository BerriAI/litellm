import { ColumnFiltersState, functionalUpdate, OnChangeFn, PaginationState, SortingState } from "@tanstack/react-table";
import { createParser, Nullable, parseAsInteger, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useMemo } from "react";

const SORT_ORDERS = ["asc", "desc"] as const;
type SortOrder = (typeof SORT_ORDERS)[number];

const STANDARD_KEYS = ["search", "sort_by", "sort_order", "page", "page_size"] as const;
type StandardKey = (typeof STANDARD_KEYS)[number];
type FilterStateKey<F extends string> = `filter_${F}`;
type StateKey<F extends string> = StandardKey | FilterStateKey<F>;

const MAX_PAGE = 100_000;
const DEFAULT_MAX_PAGE_SIZE = 100;

export interface UrlTableStateOptions<F extends string> {
  sortFields: readonly string[];
  defaultSort: { id: string; desc: boolean };
  defaultPageSize: number;
  maxPageSize?: number;
  filterColumns: readonly F[];
  keyPrefix?: string;
  urlKeys?: Partial<Record<StateKey<F>, string>>;
}

export interface UrlTableState {
  search: string;
  setSearch: (value: string) => void;
  sorting: SortingState;
  onSortingChange: OnChangeFn<SortingState>;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  columnFilters: ColumnFiltersState;
  onColumnFiltersChange: OnChangeFn<ColumnFiltersState>;
}

const boundedInteger = (min: number, max: number, fallback: number) =>
  createParser({
    parse: (value: string) => {
      const parsed = parseAsInteger.parse(value);
      return parsed === null ? null : Math.min(Math.max(parsed, min), max);
    },
    serialize: String,
  }).withDefault(fallback);

const optionalString = parseAsString.withDefault("");
type OptionalStringParser = typeof optionalString;
const sortOrderParser = (fallback: SortOrder) => parseAsStringLiteral(SORT_ORDERS).withDefault(fallback);

interface StandardValues {
  search: string;
  sort_by: string;
  sort_order: SortOrder;
  page: number;
  page_size: number;
}
type FilterValues<F extends string> = Record<FilterStateKey<F>, string>;
type StandardUpdate = Partial<Nullable<StandardValues>>;
type FilterUpdate<F extends string> = Record<FilterStateKey<F>, string | null> & Pick<Nullable<StandardValues>, "page">;
type SetTableValues<F extends string> = (update: StandardUpdate | FilterUpdate<F> | null) => Promise<URLSearchParams>;

interface TableQueryState<F extends string> {
  values: StandardValues;
  filters: FilterValues<F>;
  setValues: SetTableValues<F>;
}

type TableParsers<F extends string> = {
  search: OptionalStringParser;
  sort_by: OptionalStringParser;
  sort_order: ReturnType<typeof sortOrderParser>;
  page: ReturnType<typeof boundedInteger>;
  page_size: ReturnType<typeof boundedInteger>;
} & Record<FilterStateKey<F>, OptionalStringParser>;

const useTableQueryStates = <F extends string>(
  parsers: TableParsers<F>,
  urlKeys: Record<StateKey<F>, string>,
): TableQueryState<F> => {
  const [state, setState] = useQueryStates(parsers, { urlKeys });
  return useMemo(
    () => ({
      values: state as StandardValues,
      filters: state as FilterValues<F>,
      setValues: setState as SetTableValues<F>,
    }),
    [state, setState],
  );
};

const filterStateKey = <F extends string>(column: F): FilterStateKey<F> => `filter_${column}`;

const filterParsers = <F extends string>(filterColumns: readonly F[]) =>
  Object.fromEntries(filterColumns.map((column) => [filterStateKey(column), optionalString])) as Record<
    FilterStateKey<F>,
    OptionalStringParser
  >;

const resolveUrlKeys = <F extends string>(
  filterColumns: readonly F[],
  keyPrefix: string,
  renamed: Partial<Record<StateKey<F>, string>>,
) => {
  const stateKeys: readonly StateKey<F>[] = [
    ...STANDARD_KEYS,
    ...filterColumns.map((column) => filterStateKey(column)),
  ];
  return Object.fromEntries(stateKeys.map((key) => [key, `${keyPrefix}${renamed[key] ?? key}`])) as Record<
    StateKey<F>,
    string
  >;
};

const filterValue = (filters: ColumnFiltersState, column: string): string | null => {
  const value = filters.find((filter) => filter.id === column)?.value;
  return (typeof value === "string" ? value.trim() : "") || null;
};

const filterUpdates = <F extends string>(filterColumns: readonly F[], filters: ColumnFiltersState) =>
  Object.fromEntries(filterColumns.map((column) => [filterStateKey(column), filterValue(filters, column)])) as Record<
    FilterStateKey<F>,
    string | null
  >;

const toSortOrder = (active: SortingState[number]): SortOrder => (active.desc ? "desc" : "asc");

export function useUrlTableState<F extends string>(options: UrlTableStateOptions<F>): UrlTableState {
  const {
    sortFields,
    defaultSort,
    defaultPageSize,
    maxPageSize = DEFAULT_MAX_PAGE_SIZE,
    filterColumns,
    keyPrefix = "",
    urlKeys: renamedKeys,
  } = options;
  const defaultSortId = defaultSort.id;
  const defaultSortOrder: SortOrder = defaultSort.desc ? "desc" : "asc";

  const parsers = useMemo<TableParsers<F>>(
    () => ({
      search: optionalString,
      sort_by: parseAsString.withDefault(defaultSortId),
      sort_order: sortOrderParser(defaultSortOrder),
      page: boundedInteger(1, MAX_PAGE, 1),
      page_size: boundedInteger(1, maxPageSize, defaultPageSize),
      ...filterParsers(filterColumns),
    }),
    [defaultSortId, defaultSortOrder, defaultPageSize, maxPageSize, filterColumns],
  );
  const urlKeys = useMemo(
    () => resolveUrlKeys(filterColumns, keyPrefix, renamedKeys ?? {}),
    [filterColumns, keyPrefix, renamedKeys],
  );
  const { values, filters, setValues } = useTableQueryStates(parsers, urlKeys);

  const sortBy = sortFields.includes(values.sort_by) ? values.sort_by : defaultSortId;
  const sortDesc = values.sort_order === "desc";
  const sorting = useMemo<SortingState>(() => [{ id: sortBy, desc: sortDesc }], [sortBy, sortDesc]);

  const pagination = useMemo<PaginationState>(
    () => ({ pageIndex: values.page - 1, pageSize: values.page_size }),
    [values.page, values.page_size],
  );

  const columnFilters = useMemo<ColumnFiltersState>(
    () =>
      filterColumns.flatMap((column) => {
        const value = filters[filterStateKey(column)].trim();
        return value ? [{ id: column, value }] : [];
      }),
    [filterColumns, filters],
  );

  const setSearch = useCallback(
    (value: string) => {
      void setValues({ search: value || null, page: null });
    },
    [setValues],
  );

  const onSortingChange = useCallback<OnChangeFn<SortingState>>(
    (updaterOrValue) => {
      const active = functionalUpdate(updaterOrValue, sorting)[0];
      void setValues({
        sort_by: active?.id ?? null,
        sort_order: active ? toSortOrder(active) : null,
        page: null,
      });
    },
    [setValues, sorting],
  );

  const onPaginationChange = useCallback<OnChangeFn<PaginationState>>(
    (updaterOrValue) => {
      const next = functionalUpdate(updaterOrValue, pagination);
      void setValues({ page: next.pageIndex + 1, page_size: next.pageSize });
    },
    [pagination, setValues],
  );

  const onColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>(
    (updaterOrValue) => {
      const next = functionalUpdate(updaterOrValue, columnFilters);
      void setValues({ ...filterUpdates(filterColumns, next), page: null });
    },
    [columnFilters, filterColumns, setValues],
  );

  return useMemo<UrlTableState>(
    () => ({
      search: values.search,
      setSearch,
      sorting,
      onSortingChange,
      pagination,
      onPaginationChange,
      columnFilters,
      onColumnFiltersChange,
    }),
    [
      values.search,
      setSearch,
      sorting,
      onSortingChange,
      pagination,
      onPaginationChange,
      columnFilters,
      onColumnFiltersChange,
    ],
  );
}
