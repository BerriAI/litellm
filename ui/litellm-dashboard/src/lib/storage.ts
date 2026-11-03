import { useCallback, useMemo, useSyncExternalStore } from "react";
import type { z } from "zod";

export type StorageArea = "local" | "session";

/** A web storage slot declared once: where it lives, what it is called, what shape it holds and what to use when absent. */
export interface StorageKey<T> {
  readonly area: StorageArea;
  readonly name: string;
  readonly schema: z.ZodType<T>;
  readonly fallback: T;
}

export function storageKey<T>(area: StorageArea, name: string, schema: z.ZodType<T>, fallback: T): StorageKey<T> {
  return { area, name, schema, fallback };
}

const CHANGE_EVENT = "litellm:storage-change";

function storageOf(area: StorageArea): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return area === "local" ? window.localStorage : window.sessionStorage;
  } catch {
    return null;
  }
}

function rawOf<T>(key: StorageKey<T>): string | null {
  try {
    return storageOf(key.area)?.getItem(key.name) ?? null;
  } catch {
    return null;
  }
}

function decode<T>(key: StorageKey<T>, raw: string | null): T {
  if (raw === null) return key.fallback;
  try {
    const parsed = key.schema.safeParse(JSON.parse(raw));
    return parsed.success ? parsed.data : key.fallback;
  } catch {
    return key.fallback;
  }
}

export function readStorage<T>(key: StorageKey<T>): T {
  return decode(key, rawOf(key));
}

export function writeStorage<T>(key: StorageKey<T>, value: T): void {
  try {
    storageOf(key.area)?.setItem(key.name, JSON.stringify(value));
  } catch {
    return;
  }
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: key.name }));
}

export function clearStorage<T>(key: StorageKey<T>): void {
  try {
    storageOf(key.area)?.removeItem(key.name);
  } catch {
    return;
  }
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: key.name }));
}

function subscribeTo(name: string, onChange: () => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key === null || event.key === name) onChange();
  };
  const onLocal = (event: Event) => {
    if ((event as CustomEvent<string>).detail === name) onChange();
  };
  window.addEventListener("storage", onStorage);
  window.addEventListener(CHANGE_EVENT, onLocal);
  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(CHANGE_EVENT, onLocal);
  };
}

/** Validated web storage as React state. Server renders see the fallback; writes from any tab update every reader. */
export function useStoredValue<T>(key: StorageKey<T>): readonly [T, (value: T) => void] {
  const subscribe = useCallback((onChange: () => void) => subscribeTo(key.name, onChange), [key.name]);
  const raw = useSyncExternalStore(
    subscribe,
    () => rawOf(key),
    () => null,
  );
  const value = useMemo(() => decode(key, raw), [key, raw]);
  const write = useCallback((next: T) => writeStorage(key, next), [key]);
  return [value, write];
}
