import { groupNameByModel, modelConflictError } from "./modelOwnership";
import type { RoutingGroup } from "./types";

const groups: RoutingGroup[] = [
  { group_name: "cheap", models: ["m1", "m2"], routing_strategy: "latency-based-routing" },
  { group_name: "security", models: ["m3"], routing_strategy: "least-busy" },
];

describe("groupNameByModel", () => {
  it("maps every claimed model to its owning group", () => {
    expect(groupNameByModel(groups)).toEqual({ m1: "cheap", m2: "cheap", m3: "security" });
  });

  it("excludes the group being edited so its own models stay selectable", () => {
    expect(groupNameByModel(groups, "cheap")).toEqual({ m3: "security" });
  });
});

describe("modelConflictError", () => {
  it("passes models that no other group claims", () => {
    expect(modelConflictError(["m4"], groupNameByModel(groups, "cheap"))).toBeNull();
    expect(modelConflictError(undefined, groupNameByModel(groups))).toBeNull();
  });

  it("names every model already claimed by another group", () => {
    const error = modelConflictError(["m1", "m3", "m4"], groupNameByModel(groups));
    expect(error).toBe(
      'Each model may belong to at most one group. Already claimed: m1 (in "cheap"), m3 (in "security")',
    );
  });
});
