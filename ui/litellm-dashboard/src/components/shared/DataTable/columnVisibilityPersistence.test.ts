import { beforeEach, describe, expect, it } from "vitest";

import { columnVisibilityStorageKey, readColumnVisibility, writeColumnVisibility } from "./columnVisibilityPersistence";

beforeEach(() => {
  window.localStorage.clear();
});

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
    expect(readColumnVisibility(window.localStorage, "missing")).toBeUndefined();
  });

  it("returns undefined for corrupt JSON instead of breaking the table", () => {
    window.localStorage.setItem("k", "{not json");
    expect(readColumnVisibility(window.localStorage, "k")).toBeUndefined();
  });

  it.each([
    ["an array", '["name"]'],
    ["a string", '"name"'],
    ["null", "null"],
    ["non-boolean values", '{"name":"yes"}'],
  ])("returns undefined for %s", (_label, stored) => {
    window.localStorage.setItem("k", stored);
    expect(readColumnVisibility(window.localStorage, "k")).toBeUndefined();
  });

  it("returns an explicitly stored empty state, distinct from a missing entry", () => {
    window.localStorage.setItem("k", "{}");
    expect(readColumnVisibility(window.localStorage, "k")).toEqual({});
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
    writeColumnVisibility(window.localStorage, "k", { name: true, email: false });
    expect(readColumnVisibility(window.localStorage, "k")).toEqual({ name: true, email: false });
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
