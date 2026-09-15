import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { usePersistedColumnVisibility } from "./usePersistedColumnVisibility";

const keyFor = (tableId: string): string => `litellm_table_columns_${tableId}`;

const stored = (tableId: string): unknown => {
  const raw = localStorage.getItem(keyFor(tableId));
  return raw === null ? null : JSON.parse(raw);
};

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

  it("returns the defaults without throwing when storage is unavailable", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("SecurityError");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("QuotaExceededError");
    });

    const { result } = renderHook(() => usePersistedColumnVisibility("keys", { spend: false }));
    expect(result.current.columnVisibility).toEqual({ spend: false });

    act(() => result.current.onColumnVisibilityChange({ email: false }));
    expect(result.current.columnVisibility).toEqual({ email: false });
  });
});
