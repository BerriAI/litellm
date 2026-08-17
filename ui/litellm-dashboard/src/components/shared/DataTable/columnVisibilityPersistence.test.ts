import { describe, expect, it } from "vitest";

import { columnVisibilityStorageKey, readColumnVisibility, writeColumnVisibility } from "./columnVisibilityPersistence";

function memoryStorage(initial: Record<string, string> = {}) {
  const entries = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => {
      entries.set(key, value);
    },
    dump: () => Object.fromEntries(entries),
  };
}

describe("columnVisibilityStorageKey", () => {
  it("is order-insensitive so reordering columns in code keeps saved preferences", () => {
    expect(columnVisibilityStorageKey(["email", "name"])).toBe(columnVisibilityStorageKey(["name", "email"]));
  });

  it("differs between tables with different column sets", () => {
    expect(columnVisibilityStorageKey(["name"])).not.toBe(columnVisibilityStorageKey(["name", "email"]));
  });

  it("returns undefined when no column has an id, disabling persistence", () => {
    expect(columnVisibilityStorageKey([])).toBeUndefined();
  });
});

describe("readColumnVisibility", () => {
  it("returns undefined when nothing was stored", () => {
    expect(readColumnVisibility(memoryStorage(), "missing")).toBeUndefined();
  });

  it("returns undefined for corrupt JSON instead of breaking the table", () => {
    const storage = memoryStorage({ k: "{not json" });
    expect(readColumnVisibility(storage, "k")).toBeUndefined();
  });

  it.each([
    ["an array", '["name"]'],
    ["a string", '"name"'],
    ["null", "null"],
    ["non-boolean values", '{"name":"yes"}'],
  ])("returns undefined for %s", (_label, stored) => {
    const storage = memoryStorage({ k: stored });
    expect(readColumnVisibility(storage, "k")).toBeUndefined();
  });

  it("returns an explicitly stored empty state, distinct from a missing entry", () => {
    const storage = memoryStorage({ k: "{}" });
    expect(readColumnVisibility(storage, "k")).toEqual({});
  });

  it("returns undefined when storage access throws", () => {
    const storage = {
      getItem: () => {
        throw new Error("denied");
      },
    };
    expect(readColumnVisibility(storage, "k")).toBeUndefined();
  });
});

describe("writeColumnVisibility", () => {
  it("round-trips a visibility state through storage", () => {
    const storage = memoryStorage();
    writeColumnVisibility(storage, "k", { name: true, email: false });
    expect(readColumnVisibility(storage, "k")).toEqual({ name: true, email: false });
  });

  it("does not throw when storage rejects the write", () => {
    const storage = {
      setItem: () => {
        throw new Error("quota");
      },
    };
    expect(() => writeColumnVisibility(storage, "k", { email: false })).not.toThrow();
  });
});
