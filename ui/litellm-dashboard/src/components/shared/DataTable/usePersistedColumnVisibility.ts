import type { OnChangeFn, VisibilityState } from "@tanstack/react-table";
import { useCallback, useState } from "react";

import { getLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";

const STORAGE_KEY_PREFIX = "litellm_table_columns_";

const EMPTY_VISIBILITY: VisibilityState = {};

function storageKey(tableId: string): string {
  return `${STORAGE_KEY_PREFIX}${tableId}`;
}

function isVisibilityState(value: unknown): value is VisibilityState {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  return Object.values(value).every((visible) => typeof visible === "boolean");
}

function readStoredVisibility(tableId: string, defaults: VisibilityState): VisibilityState {
  const raw = getLocalStorageItem(storageKey(tableId));
  if (raw === null) {
    return defaults;
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    return isVisibilityState(parsed) ? { ...defaults, ...parsed } : defaults;
  } catch {
    return defaults;
  }
}

export function usePersistedColumnVisibility(
  tableId: string,
  defaults: VisibilityState = EMPTY_VISIBILITY,
): { columnVisibility: VisibilityState; onColumnVisibilityChange: OnChangeFn<VisibilityState> } {
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(() =>
    readStoredVisibility(tableId, defaults),
  );

  const onColumnVisibilityChange = useCallback<OnChangeFn<VisibilityState>>(
    (updater) => {
      setColumnVisibility((previous) => {
        const next = typeof updater === "function" ? updater(previous) : updater;
        setLocalStorageItem(storageKey(tableId), JSON.stringify(next));
        return next;
      });
    },
    [tableId],
  );

  return { columnVisibility, onColumnVisibilityChange };
}
