import { describe, expect, it } from "vitest";

import { modelSentinelOptions } from "./keyEditFieldNormalizers";

describe("modelSentinelOptions", () => {
  it("offers No Default Models on a key with no team, so an access-group-only key is selectable", () => {
    expect(modelSentinelOptions(null, false)).toEqual([
      { value: "all-proxy-models", label: "All Proxy Models" },
      { value: "no-default-models", label: "No Default Models" },
    ]);
  });

  it("offers No Default Models on a team key once the team has loaded", () => {
    expect(modelSentinelOptions("team-1", true)).toEqual([
      { value: "all-team-models", label: "All Team Models" },
      { value: "no-default-models", label: "No Default Models" },
    ]);
  });

  it("offers nothing while a team key is still loading its team", () => {
    expect(modelSentinelOptions("team-1", false)).toEqual([]);
  });
});
