import { describe, expect, it } from "vitest";
import { searchToolPermissionCopy } from "./searchToolPermissionCopy";

describe("searchToolPermissionCopy", () => {
  it("says an empty list allows every search tool when default deny is off", () => {
    const copy = searchToolPermissionCopy(false);
    expect(copy.hint).toContain("Leave empty to allow all search tools");
    expect(copy.placeholder).toBe("Select search tools (optional, empty = all allowed)");
    expect(copy.emptyState).toContain("all configured search tools are allowed");
  });

  it("says an empty list denies every search tool when default deny is on", () => {
    const copy = searchToolPermissionCopy(true);
    expect(copy.hint).toContain("leaving this empty denies every search tool");
    expect(copy.placeholder).toBe("Select search tools (empty = none allowed)");
    expect(copy.emptyState).toContain("no search tool is allowed");
    expect(Object.values(copy).join(" ")).not.toContain("all allowed");
  });
});
