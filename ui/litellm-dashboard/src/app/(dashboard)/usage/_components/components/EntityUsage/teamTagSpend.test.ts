import { describe, expect, it } from "vitest";

import { singleTeamId } from "./teamTagSpend";

describe("singleTeamId", () => {
  it("is the team only when exactly one is selected", () => {
    expect(singleTeamId(["team-a"])).toBe("team-a");
    expect(singleTeamId([])).toBeNull();
    expect(singleTeamId(["team-a", "team-b"])).toBeNull();
  });
});
