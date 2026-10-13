import { describe, expect, it } from "vitest";

import { buildAccessGroupCreateBody, emptyAccessGroupFormValues } from "./mapper";

describe("buildAccessGroupCreateBody", () => {
  it("sends only the trimmed name for a minimal create", () => {
    const values = { ...emptyAccessGroupFormValues, name: "  prod models  " };
    const body = buildAccessGroupCreateBody(values);
    expect(body).toStrictEqual({ access_group_name: "prod models" });
  });

  it("omits a whitespace-only description", () => {
    const values = { ...emptyAccessGroupFormValues, name: "g", description: "   " };
    const body = buildAccessGroupCreateBody(values);
    expect(body).toStrictEqual({ access_group_name: "g" });
  });

  it("maps every populated field into the request body", () => {
    const values = {
      name: "g",
      description: " engineering access ",
      modelIds: ["gpt-5.2"],
      mcpServerIds: ["srv-1", "srv-2"],
      agentIds: ["agent-1"],
      accessRoutes: ["/global/spend/report"],
    };
    const body = buildAccessGroupCreateBody(values);
    const expectedBody = {
      access_group_name: "g",
      description: "engineering access",
      access_model_names: ["gpt-5.2"],
      access_mcp_server_ids: ["srv-1", "srv-2"],
      access_agent_ids: ["agent-1"],
      access_routes: ["/global/spend/report"],
    };
    expect(body).toStrictEqual(expectedBody);
  });
});
