import { functionalUpdate, type OnChangeFn, type SortingState } from "@tanstack/react-table";
import { useCallback, useMemo } from "react";

import { useUrlTableState, type UrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";

const UNSORTED_COLUMN_ID = "";

const HEALTH_TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: ["model_id", "model_name", "team_id", "health_status", "last_check", "last_success"],
  defaultSort: { id: UNSORTED_COLUMN_ID, desc: false },
  defaultPageSize: 50,
  filterColumns: [],
  keyPrefix: "health_",
};

export function useHealthTableUrlState(): UrlTableState {
  const state = useUrlTableState(HEALTH_TABLE_STATE_OPTIONS);
  const { sorting: urlSorting, onSortingChange: setUrlSorting, pagination, onPaginationChange } = state;

  const sorting = useMemo<SortingState>(
    () => urlSorting.filter((entry) => entry.id !== UNSORTED_COLUMN_ID),
    [urlSorting],
  );

  const onSortingChange = useCallback<OnChangeFn<SortingState>>(
    (updater) => {
      const pageBeforeSort = pagination;
      setUrlSorting(functionalUpdate(updater, sorting));
      onPaginationChange(pageBeforeSort);
    },
    [setUrlSorting, sorting, onPaginationChange, pagination],
  );

  return useMemo(() => ({ ...state, sorting, onSortingChange }), [state, sorting, onSortingChange]);
}
