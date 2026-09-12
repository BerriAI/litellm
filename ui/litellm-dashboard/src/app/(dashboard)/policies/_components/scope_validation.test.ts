import { describe, expect, it } from "vitest";
import { getInvalidTeamEntries, isWildcardPattern } from "./scope_validation";

describe("isWildcardPattern", () => {
  it("treats a trailing * as a wildcard", () => {
    expect(isWildcardPattern("healthcare-*")).toBe(true);
    expect(isWildcardPattern("*")).toBe(true);
  });

  it("treats ?, a non-trailing *, and plain aliases as concrete (matches request-time semantics)", () => {
    expect(isWildcardPattern("team-?")).toBe(false);
    expect(isWildcardPattern("heal*care")).toBe(false);
    expect(isWildcardPattern("payments-team")).toBe(false);
  });
});

describe("getInvalidTeamEntries", () => {
  const availableTeams = ["payments-team", "healthcare-us"];

  it("flags a concrete alias that is not an existing team", () => {
    expect(getInvalidTeamEntries(["payments-team", "ghost-team"], availableTeams)).toEqual(["ghost-team"]);
  });

  it("accepts an existing team alias", () => {
    expect(getInvalidTeamEntries(["payments-team"], availableTeams)).toEqual([]);
  });

  it("accepts a wildcard pattern even when it currently matches no team", () => {
    expect(getInvalidTeamEntries(["healthcare-*", "brand-new-*"], availableTeams)).toEqual([]);
  });

  it("returns nothing for an empty selection", () => {
    expect(getInvalidTeamEntries([], availableTeams)).toEqual([]);
  });
});
