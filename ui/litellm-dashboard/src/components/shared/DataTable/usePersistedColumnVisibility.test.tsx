import type { VisibilityState } from "@tanstack/react-table";
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { usePersistedColumnVisibility } from "./usePersistedColumnVisibility";

const keyFor = (tableId: string): string => `litellm_table_columns_${tableId}`;

const stored = (tableId: string): unknown => {
  const raw = localStorage.getItem(keyFor(tableId));
  return raw === null ? null : JSON.parse(raw);
};

const showEveryColumn = (previous: VisibilityState): VisibilityState =>
  Object.fromEntries(Object.keys(previous).map((column) => [column, true]));

describe("usePersistedColumnVisibility", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("layers the stored choices over the defaults, so a default added after the snapshot still applies", () => {
    localStorage.setItem(keyFor("keys"), JSON.stringify({ email: false, spend: true }));

    const { result } = renderHook(() => usePersistedColumnVisibility("keys", { spend: false, name: false }));

    expect(result.current.columnVisibility).toEqual({ email: false, spend: true, name: false });
  });

  it("falls back to the defaults when nothing is stored, and to {} without defaults", () => {
    const withDefaults = renderHook(() => usePersistedColumnVisibility("keys", { spend: false }));
    expect(withDefaults.result.current.columnVisibility).toEqual({ spend: false });

    const bare = renderHook(() => usePersistedColumnVisibility("keys"));
    expect(bare.result.current.columnVisibility).toEqual({});
  });

  it("writes an object update to state and storage", () => {
    const { result } = renderHook(() => usePersistedColumnVisibility("keys"));

    act(() => result.current.onColumnVisibilityChange({ email: false }));

    expect(result.current.columnVisibility).toEqual({ email: false });
    expect(stored("keys")).toEqual({ email: false });
  });

  it("resolves a function updater against the current state before persisting", () => {
    localStorage.setItem(keyFor("keys"), JSON.stringify({ email: false }));
    const { result } = renderHook(() => usePersistedColumnVisibility("keys"));

    act(() => result.current.onColumnVisibilityChange((previous) => ({ ...previous, name: false })));

    expect(result.current.columnVisibility).toEqual({ email: false, name: false });
    expect(stored("keys")).toEqual({ email: false, name: false });
  });

  it("hands a function updater the default-hidden columns, so showing every column sticks", () => {
    const { result } = renderHook(() => usePersistedColumnVisibility("keys", { spend: false }));

    act(() => result.current.onColumnVisibilityChange(showEveryColumn));

    expect(result.current.columnVisibility).toEqual({ spend: true });
    expect(stored("keys")).toEqual({ spend: true });
  });

  it.each([
    ["truncated JSON", '{"email":fal'],
    ["a JSON scalar", "42"],
    ["a JSON array", "[true]"],
    ["non-boolean values", JSON.stringify({ email: "no" })],
  ])("falls back to the defaults when storage holds %s", (_label, raw) => {
    localStorage.setItem(keyFor("keys"), raw);

    const { result } = renderHook(() => usePersistedColumnVisibility("keys", { spend: false }));

    expect(result.current.columnVisibility).toEqual({ spend: false });
  });

  it("keeps distinct tableIds isolated in state and storage", () => {
    const keys = renderHook(() => usePersistedColumnVisibility("keys"));
    const teams = renderHook(() => usePersistedColumnVisibility("teams"));

    act(() => keys.result.current.onColumnVisibilityChange({ email: false }));

    expect(keys.result.current.columnVisibility).toEqual({ email: false });
    expect(teams.result.current.columnVisibility).toEqual({});
    expect(stored("keys")).toEqual({ email: false });
    expect(stored("teams")).toBeNull();
  });

  it("reads and writes the new table's columns after the tableId changes", () => {
    localStorage.setItem(keyFor("keys"), JSON.stringify({ email: false }));
    localStorage.setItem(keyFor("teams"), JSON.stringify({ spend: false }));
    const { result, rerender } = renderHook(({ tableId }) => usePersistedColumnVisibility(tableId), {
      initialProps: { tableId: "keys" },
    });

    rerender({ tableId: "teams" });
    expect(result.current.columnVisibility).toEqual({ spend: false });

    act(() => result.current.onColumnVisibilityChange((previous) => ({ ...previous, name: false })));
    expect(stored("teams")).toEqual({ spend: false, name: false });
    expect(stored("keys")).toEqual({ email: false });
  });

  it("applies new defaults passed after mount", () => {
    const { result, rerender } = renderHook(({ defaults }) => usePersistedColumnVisibility("keys", defaults), {
      initialProps: { defaults: { spend: false } },
    });

    rerender({ defaults: { name: false } });

    expect(result.current.columnVisibility).toEqual({ name: false });
  });

  it("shows a change another tab saved for the same table", () => {
    const { result } = renderHook(() => usePersistedColumnVisibility("keys"));

    act(() => {
      localStorage.setItem(keyFor("keys"), JSON.stringify({ email: false }));
      window.dispatchEvent(new StorageEvent("storage", { key: keyFor("keys") }));
    });

    expect(result.current.columnVisibility).toEqual({ email: false });
  });

  it("keeps a toggle that storage refused, and saves the next one once storage accepts it", () => {
    localStorage.setItem(keyFor("full"), JSON.stringify({ spend: false }));
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(Storage.prototype, "setItem").mockImplementationOnce(() => {
      throw new Error("QuotaExceededError");
    });
    const { result } = renderHook(() => usePersistedColumnVisibility("full"));

    act(() => result.current.onColumnVisibilityChange({ email: false }));
    expect(result.current.columnVisibility).toEqual({ email: false });
    expect(stored("full")).toEqual({ spend: false });

    act(() => result.current.onColumnVisibilityChange({ name: false }));
    expect(result.current.columnVisibility).toEqual({ name: false });
    expect(stored("full")).toEqual({ name: false });
  });

  it("returns the defaults without throwing when storage is unavailable", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("SecurityError");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("SecurityError");
    });

    const { result } = renderHook(() => usePersistedColumnVisibility("blocked", { spend: false }));
    expect(result.current.columnVisibility).toEqual({ spend: false });

    act(() => result.current.onColumnVisibilityChange((previous) => ({ ...previous, email: false })));
    expect(result.current.columnVisibility).toEqual({ spend: false, email: false });
  });
});
