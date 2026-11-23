import { describe, expect, it } from "vitest";
import { appAccessChanged, grantedAppIds } from "./appAccessApi";

describe("internal access", () => {
  it("reads the granted apps and treats a missing object permission as no apps", () => {
    expect(grantedAppIds({ object_permission: { apps: ["chat", "search"] } })).toEqual(["chat", "search"]);
    expect(grantedAppIds({ object_permission: null })).toEqual([]);
    expect(grantedAppIds(undefined)).toEqual([]);
  });

  it("only reports a change when the set of apps differs, whatever the order", () => {
    expect(appAccessChanged(["chat", "search"], ["search", "chat"])).toBe(false);
    expect(appAccessChanged(["chat"], ["chat", "search"])).toBe(true);
    expect(appAccessChanged(["chat"], [])).toBe(true);
  });
});
