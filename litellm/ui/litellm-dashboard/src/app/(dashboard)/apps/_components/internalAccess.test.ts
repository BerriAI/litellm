import { describe, expect, it } from "vitest";
import { appAccessChanged, teamAppIds } from "./internalAccess";

describe("internal access", () => {
  it("reads a team's granted apps and treats a team with no object permission as having none", () => {
    expect(teamAppIds({ team_info: { object_permission: { apps: ["chat", "search"] } } })).toEqual(["chat", "search"]);
    expect(teamAppIds({ team_info: { object_permission: null } })).toEqual([]);
    expect(teamAppIds(undefined)).toEqual([]);
  });

  it("only reports a change when the set of apps differs, whatever the order", () => {
    expect(appAccessChanged(["chat", "search"], ["search", "chat"])).toBe(false);
    expect(appAccessChanged(["chat"], ["chat", "search"])).toBe(true);
    expect(appAccessChanged(["chat"], [])).toBe(true);
  });
});
