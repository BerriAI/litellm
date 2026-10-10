import { describe, expect, it } from "vitest";

import { idsBetween, paintRowSelection } from "./rowSelectionRange";

const ids = ["a", "b", "c", "d", "e"];

describe("idsBetween", () => {
  it("returns the inclusive range in table order whichever way the pointer moved", () => {
    expect(idsBetween(ids, "b", "d")).toEqual(["b", "c", "d"]);
    expect(idsBetween(ids, "d", "b")).toEqual(["b", "c", "d"]);
    expect(idsBetween(ids, "c", "c")).toEqual(["c"]);
  });

  it("returns nothing when either end is not on the page", () => {
    expect(idsBetween(ids, "e", "z")).toEqual([]);
    expect(idsBetween(ids, "z", "e")).toEqual([]);
  });
});

describe("paintRowSelection", () => {
  it("selects the range and keeps selections outside it, including rows on other pages", () => {
    const painted = { a: true, offpage: true, c: true, d: true };
    expect(paintRowSelection({ a: true, offpage: true }, ["c", "d"], true)).toEqual(painted);
  });

  it("deselects the range and keeps selections outside it", () => {
    const snapshot = { a: true, b: true, c: true, e: true };
    expect(paintRowSelection(snapshot, ["b", "c", "d"], false)).toEqual({ a: true, e: true });
  });

  it("restores the pre-drag state for rows the range no longer covers", () => {
    const snapshot = { b: true };
    expect(paintRowSelection(snapshot, ["c", "d", "e"], false)).toEqual({ b: true });
    expect(paintRowSelection(snapshot, ["c"], false)).toEqual({ b: true });
    expect(snapshot).toEqual({ b: true });
  });
});
