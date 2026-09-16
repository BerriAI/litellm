import { useUrlTableState, type UrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { parseAsString, useQueryStates } from "nuqs";
import { useCallback } from "react";

export const PROJECTS_DEFAULT_PAGE_SIZE = 10;
export const PROJECT_KEYS_DEFAULT_PAGE_SIZE = 5;

const PROJECT_KEYS_URL_PREFIX = "keys_";
const TABLE_STATE_URL_KEYS = ["search", "sort_by", "sort_order", "page", "page_size"] as const;

const PROJECTS_TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: [],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: PROJECTS_DEFAULT_PAGE_SIZE,
  filterColumns: [],
  urlKeys: { search: "project_search" },
};

const PROJECT_KEYS_TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: [],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: PROJECT_KEYS_DEFAULT_PAGE_SIZE,
  maxPageSize: 25,
  filterColumns: [],
  keyPrefix: PROJECT_KEYS_URL_PREFIX,
};

const PROJECT_KEYS_URL_STATE = Object.fromEntries(
  TABLE_STATE_URL_KEYS.map((key) => [`${PROJECT_KEYS_URL_PREFIX}${key}`, parseAsString]),
);

export const useProjectsTableState = (): UrlTableState => useUrlTableState(PROJECTS_TABLE_STATE_OPTIONS);

export const useProjectKeysTableState = (): UrlTableState => useUrlTableState(PROJECT_KEYS_TABLE_STATE_OPTIONS);

export function useClearProjectKeysTableState(): () => void {
  const [, setProjectKeysUrlState] = useQueryStates(PROJECT_KEYS_URL_STATE);
  return useCallback(() => void setProjectKeysUrlState(null), [setProjectKeysUrlState]);
}
