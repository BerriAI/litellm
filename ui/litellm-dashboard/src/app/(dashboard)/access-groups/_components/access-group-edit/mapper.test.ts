import { describe, expect, it } from "vitest";

import type { AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";

import { buildAccessGroupPatchBody, formValuesFromAccessGroup } from "./mapper";

const GROUP: AccessGroupResponse = {
  access_group_id: "ag-1",
  access_group_name: "prod-models",
  description: null,
  access_model_names: ["gpt-5.2"],
  access_mcp_server_ids: [],
  access_agent_ids: ["agent-1"],
  assigned_team_ids: [],
  assigned_key_ids: [],
  created_at: "2026-01-01T00:00:00Z",
  created_by: "admin",
  updated_at: "2026-01-01T00:00:00Z",
  updated_by: "admin",
};

describe("formValuesFromAccessGroup", () => {
  it("hydrates every field and turns a null description into an empty string", () => {
    const expected = {
      name: "prod-models",
      description: "",
      modelIds: ["gpt-5.2"],
      mcpServerIds: [],
      agentIds: ["agent-1"],
    };
    expect(formValuesFromAccessGroup(GROUP)).toStrictEqual(expected);
  });
});

describe("buildAccessGroupPatchBody", () => {
  it("sends nothing when nothing is dirty", () => {
    expect(buildAccessGroupPatchBody({})).toStrictEqual({});
  });

  it("maps only the dirty fields", () => {
    expect(buildAccessGroupPatchBody({ name: "  renamed  ", agentIds: [] })).toStrictEqual({
      access_group_name: "renamed",
      access_agent_ids: [],
    });
  });

  it("clears the description with null when it is blanked out", () => {
    expect(buildAccessGroupPatchBody({ description: "   " })).toStrictEqual({ description: null });
  });

  it("sends an emptied list as [] so the grant is removed rather than left untouched", () => {
    expect(buildAccessGroupPatchBody({ modelIds: [], mcpServerIds: ["srv-1"] })).toStrictEqual({
      access_model_names: [],
      access_mcp_server_ids: ["srv-1"],
    });
  });
});
