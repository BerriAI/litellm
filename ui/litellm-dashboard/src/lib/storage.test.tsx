import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { z } from "zod";

import { clearStorage, readStorage, storageKey, useStoredValue, writeStorage } from "./storage";

const flag = storageKey("local", "test.flag", z.boolean(), false);
const prefs = storageKey("session", "test.prefs", z.object({ theme: z.enum(["light", "dark"]) }), { theme: "light" });

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
});

describe("readStorage", () => {
  it("returns the fallback when the slot is empty, unparsable, or fails validation", () => {
    expect(readStorage(flag)).toBe(false);
    window.localStorage.setItem(flag.name, "{not json");
    expect(readStorage(flag)).toBe(false);
    window.localStorage.setItem(flag.name, JSON.stringify("yes"));
    expect(readStorage(flag)).toBe(false);
    window.sessionStorage.setItem(prefs.name, JSON.stringify({ theme: "sepia" }));
    expect(readStorage(prefs)).toEqual({ theme: "light" });
  });

  it("round-trips a written value through its own area only", () => {
    writeStorage(flag, true);
    writeStorage(prefs, { theme: "dark" });
    expect(readStorage(flag)).toBe(true);
    expect(readStorage(prefs)).toEqual({ theme: "dark" });
    expect(window.sessionStorage.getItem(flag.name)).toBeNull();
    expect(window.localStorage.getItem(prefs.name)).toBeNull();
    clearStorage(flag);
    expect(readStorage(flag)).toBe(false);
  });
});

describe("useStoredValue", () => {
  it("starts from what is stored and re-renders readers when the slot changes", () => {
    writeStorage(prefs, { theme: "dark" });
    const { result } = renderHook(() => useStoredValue(prefs));
    expect(result.current[0]).toEqual({ theme: "dark" });
    act(() => result.current[1]({ theme: "light" }));
    expect(result.current[0]).toEqual({ theme: "light" });
    expect(readStorage(prefs)).toEqual({ theme: "light" });
    act(() => clearStorage(prefs));
    expect(result.current[0]).toEqual(prefs.fallback);
  });

  it("follows writes made in another tab", () => {
    const { result } = renderHook(() => useStoredValue(flag));
    expect(result.current[0]).toBe(false);
    act(() => {
      window.localStorage.setItem(flag.name, "true");
      window.dispatchEvent(new StorageEvent("storage", { key: flag.name }));
    });
    expect(result.current[0]).toBe(true);
  });
});
