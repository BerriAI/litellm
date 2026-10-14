import { describe, expect, it } from "vitest";

import { buildAccessGroupCreateBody, emptyAccessGroupFormValues } from "./mapper";

describe("buildAccessGroupCreateBody", () => {
  it("sends only the trimmed name for a minimal create", () => {
    expect(buildAccessGroupCreateBody({ ...emptyAccessGroupFormValues, name: "  prod models  " })).toStrictEqual({
      access_group_name: "prod models",
    });
  });

  it("omits a whitespace-only description", () => {
    expect(buildAccessGroupCreateBody({ ...emptyAccessGroupFormValues, name: "g", description: "   " })).toStrictEqual({
      access_group_name: "g",
    });
  });

  it("maps every populated field into the request body", () => {
    expect(
      buildAccessGroupCreateBody({
        name: "g",
        description: " engineering access ",
        modelIds: ["gpt-5.2"],
        mcpServerIds: ["srv-1", "srv-2"],
        mcpToolPermissions: { "srv-1": ["read_file"], "srv-2": [] },
        agentIds: ["agent-1"],
      }),
    ).toStrictEqual({
      access_group_name: "g",
      description: "engineering access",
      access_model_names: ["gpt-5.2"],
      access_mcp_server_ids: ["srv-1", "srv-2"],
      mcp_tool_permissions: { "srv-1": ["read_file"], "srv-2": [] },
      access_agent_ids: ["agent-1"],
    });
  });

  it("omits tool permissions when no server is selected", () => {
    expect(
      buildAccessGroupCreateBody({
        ...emptyAccessGroupFormValues,
        name: "g",
        mcpToolPermissions: { "srv-1": ["read_file"] },
      }),
    ).toStrictEqual({ access_group_name: "g" });
  });

  it("omits an empty tool permission map so servers stay unrestricted", () => {
    expect(
      buildAccessGroupCreateBody({ ...emptyAccessGroupFormValues, name: "g", mcpServerIds: ["srv-1"] }),
    ).toStrictEqual({ access_group_name: "g", access_mcp_server_ids: ["srv-1"] });
  });
});
