import type { OnChangeFn, VisibilityState } from "@tanstack/react-table";
import { useCallback, useMemo, useSyncExternalStore } from "react";

import {
  LOCAL_STORAGE_EVENT,
  emitLocalStorageChange,
  getLocalStorageItem,
  setLocalStorageItem,
} from "@/utils/localStorageUtils";

const STORAGE_KEY_PREFIX = "litellm_table_columns_";

const EMPTY_VISIBILITY: VisibilityState = {};

const unsavedWrites = new Map<string, string>();

function storageKey(tableId: string): string {
  return `${STORAGE_KEY_PREFIX}${tableId}`;
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener("storage", onChange);
  window.addEventListener(LOCAL_STORAGE_EVENT, onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener(LOCAL_STORAGE_EVENT, onChange);
  };
}

function readRaw(key: string): string | null {
  return unsavedWrites.get(key) ?? getLocalStorageItem(key);
}

function writeRaw(key: string, raw: string): void {
  setLocalStorageItem(key, raw);
  if (getLocalStorageItem(key) === raw) {
    unsavedWrites.delete(key);
  } else {
    unsavedWrites.set(key, raw);
  }
  emitLocalStorageChange(key);
}

function isVisibilityState(value: unknown): value is VisibilityState {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  return Object.values(value).every((visible) => typeof visible === "boolean");
}

function parseVisibility(raw: string | null, defaults: VisibilityState): VisibilityState {
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
  const key = storageKey(tableId);
  const raw = useSyncExternalStore(
    subscribe,
    () => readRaw(key),
    () => null,
  );
  const columnVisibility = useMemo(() => parseVisibility(raw, defaults), [raw, defaults]);

  const onColumnVisibilityChange = useCallback<OnChangeFn<VisibilityState>>(
    (updater) => {
      const next = typeof updater === "function" ? updater(parseVisibility(readRaw(key), defaults)) : updater;
      writeRaw(key, JSON.stringify(next));
    },
    [key, defaults],
  );

  return { columnVisibility, onColumnVisibilityChange };
}
