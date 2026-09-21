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
        agentIds: ["agent-1"],
      }),
    ).toStrictEqual({
      access_group_name: "g",
      description: "engineering access",
      access_model_names: ["gpt-5.2"],
      access_mcp_server_ids: ["srv-1", "srv-2"],
      access_agent_ids: ["agent-1"],
    });
  });
});
