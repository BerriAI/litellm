"use client";

import type { ColumnDef, OnChangeFn, RowData, VisibilityState } from "@tanstack/react-table";
import { useEffect, useMemo, useState } from "react";

import { getLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";

function decode(raw: string): unknown {
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

export function parseColumnVisibility(raw: string | null, columnIds: readonly string[]): VisibilityState {
  const stored = raw === null ? null : asRecord(decode(raw));
  if (stored === null) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(stored).filter(
      (entry): entry is [string, boolean] => columnIds.includes(entry[0]) && typeof entry[1] === "boolean",
    ),
  );
}

export function useColumnVisibilityPreference<TData extends RowData, TValue>(
  storageKey: string,
  columns: readonly ColumnDef<TData, TValue>[],
): readonly [VisibilityState, OnChangeFn<VisibilityState>] {
  const columnIds = useMemo(() => columns.flatMap((column) => (column.id === undefined ? [] : [column.id])), [columns]);
  const [visibility, setVisibility] = useState<VisibilityState>(() =>
    parseColumnVisibility(getLocalStorageItem(storageKey), columnIds),
  );

  useEffect(() => {
    setLocalStorageItem(storageKey, JSON.stringify(visibility));
  }, [storageKey, visibility]);

  return [visibility, setVisibility] as const;
}
