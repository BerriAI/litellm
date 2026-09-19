import { describe, expect, it } from "vitest";
import { computeInheritedGrants, inheritedGrantTooltip } from "./inheritedGrants";
import { TeamAccessGroupModelGrant } from "../team/teamModelAccess";

const GRANTS: TeamAccessGroupModelGrant[] = [
  { access_group_id: "ag-1", access_group_name: "platform-tools", models: [], mcp_server_ids: ["mcp-1", "mcp-2"] },
  {
    access_group_id: "ag-2",
    access_group_name: "support",
    models: [],
    mcp_server_ids: ["mcp-2"],
    agent_ids: ["agent-1"],
  },
];

describe("computeInheritedGrants", () => {
  it("attributes each id to every group that grants it, in group order", () => {
    expect(computeInheritedGrants(["mcp-1", "mcp-2"], GRANTS, (g) => g.mcp_server_ids)).toEqual([
      { id: "mcp-1", accessGroupNames: ["platform-tools"] },
      { id: "mcp-2", accessGroupNames: ["platform-tools", "support"] },
    ]);
  });

  it("keeps ids the flat list carries but no group detail explains, with no group names", () => {
    expect(computeInheritedGrants(["agent-1", "agent-legacy"], GRANTS, (g) => g.agent_ids)).toEqual([
      { id: "agent-1", accessGroupNames: ["support"] },
      { id: "agent-legacy", accessGroupNames: [] },
    ]);
  });

  it("falls back to the group details when the flat list is missing, without duplicates", () => {
    expect(computeInheritedGrants(undefined, GRANTS, (g) => g.mcp_server_ids).map((g) => g.id)).toEqual([
      "mcp-1",
      "mcp-2",
    ]);
  });

  it("returns nothing when neither source has ids", () => {
    expect(computeInheritedGrants(undefined, undefined, (g) => g.agent_ids)).toEqual([]);
  });
});

describe("inheritedGrantTooltip", () => {
  it("names a single group", () => {
    expect(inheritedGrantTooltip({ id: "mcp-1", accessGroupNames: ["platform-tools"] })).toBe(
      "Granted via access group platform-tools. Full ID: mcp-1",
    );
  });

  it("lists several groups", () => {
    expect(inheritedGrantTooltip({ id: "mcp-2", accessGroupNames: ["platform-tools", "support"] })).toBe(
      "Granted via access groups platform-tools, support. Full ID: mcp-2",
    );
  });

  it("stays generic when the proxy did not say which group granted it", () => {
    expect(inheritedGrantTooltip({ id: "agent-legacy", accessGroupNames: [] })).toBe(
      "Granted via an access group. Full ID: agent-legacy",
    );
  });
});
