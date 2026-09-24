import { functionalUpdate, type OnChangeFn, type PaginationState } from "@tanstack/react-table";
import { useUrlTableState, type UrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { parseAsInteger, useQueryStates } from "nuqs";
import { useCallback, useMemo } from "react";

export const PROJECTS_DEFAULT_PAGE_SIZE = 10;
export const PROJECT_KEYS_DEFAULT_PAGE_SIZE = 5;
export const PROJECT_KEYS_PAGE_SIZE_OPTIONS = [PROJECT_KEYS_DEFAULT_PAGE_SIZE, 10, 25];

const PROJECTS_TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: [],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: PROJECTS_DEFAULT_PAGE_SIZE,
  filterColumns: [],
  urlKeys: { search: "project_search" },
};

const PROJECTS_PAGE_PARAMS = {
  page: parseAsInteger.withDefault(1),
  page_size: parseAsInteger.withDefault(PROJECTS_DEFAULT_PAGE_SIZE),
};

const PROJECT_KEYS_TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: [],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: PROJECT_KEYS_DEFAULT_PAGE_SIZE,
  maxPageSize: Math.max(...PROJECT_KEYS_PAGE_SIZE_OPTIONS),
  filterColumns: [],
  keyPrefix: "keys_",
};

export function useProjectsTableState(): UrlTableState {
  const tableState = useUrlTableState(PROJECTS_TABLE_STATE_OPTIONS);
  const [, setPageParams] = useQueryStates(PROJECTS_PAGE_PARAMS, { history: "push" });
  const { pagination } = tableState;

  const onPaginationChange = useCallback<OnChangeFn<PaginationState>>(
    (updaterOrValue) => {
      const next = functionalUpdate(updaterOrValue, pagination);
      void setPageParams({ page: next.pageIndex + 1, page_size: next.pageSize });
    },
    [pagination, setPageParams],
  );

  return useMemo(() => ({ ...tableState, onPaginationChange }), [tableState, onPaginationChange]);
}

export function useProjectKeysTableState(): UrlTableState {
  const tableState = useUrlTableState(PROJECT_KEYS_TABLE_STATE_OPTIONS);
  const { pagination: urlPagination, onPaginationChange: writePagination } = tableState;
  const pageSize = PROJECT_KEYS_PAGE_SIZE_OPTIONS.includes(urlPagination.pageSize)
    ? urlPagination.pageSize
    : PROJECT_KEYS_DEFAULT_PAGE_SIZE;

  const pagination = useMemo<PaginationState>(
    () => ({ pageIndex: urlPagination.pageIndex, pageSize }),
    [urlPagination.pageIndex, pageSize],
  );

  const onPaginationChange = useCallback<OnChangeFn<PaginationState>>(
    (updaterOrValue) => writePagination(functionalUpdate(updaterOrValue, pagination)),
    [pagination, writePagination],
  );

  return useMemo(
    () => ({ ...tableState, pagination, onPaginationChange }),
    [tableState, pagination, onPaginationChange],
  );
}

export function useClearProjectKeysTableState(): () => void {
  const { setSearch, onSortingChange, onColumnFiltersChange, onPaginationChange } = useProjectKeysTableState();
  return useCallback(() => {
    setSearch("");
    onSortingChange([]);
    onColumnFiltersChange([]);
    onPaginationChange({ pageIndex: 0, pageSize: PROJECT_KEYS_DEFAULT_PAGE_SIZE });
  }, [setSearch, onSortingChange, onColumnFiltersChange, onPaginationChange]);
}
