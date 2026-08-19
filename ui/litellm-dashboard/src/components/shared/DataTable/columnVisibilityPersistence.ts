import type { VisibilityState } from "@tanstack/react-table";

const STORAGE_KEY_PREFIX = "litellm.dataTable.columnVisibility:";

export function columnVisibilityStorageKey(columnIds: readonly string[]): string | undefined {
  if (columnIds.length === 0) {
    return undefined;
  }
  return STORAGE_KEY_PREFIX + [...columnIds].sort().join(",");
}

export function readColumnVisibility(storage: Pick<Storage, "getItem">, key: string): VisibilityState | undefined {
  try {
    const raw = storage.getItem(key);
    if (raw === null) {
      return undefined;
    }
    const parsed: unknown = JSON.parse(raw);
    if (!isVisibilityState(parsed)) {
      return undefined;
    }
    return parsed;
  } catch {
    return undefined;
  }
}

export function writeColumnVisibility(
  storage: Pick<Storage, "setItem">,
  key: string,
  visibility: VisibilityState,
): void {
  try {
    storage.setItem(key, JSON.stringify(visibility));
  } catch {
    return;
  }
}

function isVisibilityState(value: unknown): value is VisibilityState {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  return Object.values(value).every((entry) => typeof entry === "boolean");
}
