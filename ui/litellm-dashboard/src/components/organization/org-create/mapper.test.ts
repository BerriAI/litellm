import { describe, expect, it } from "vitest";

import { buildOrgCreateBody, emptyOrgFormValues } from "./mapper";

describe("buildOrgCreateBody", () => {
  it("sends only alias and models for a minimal form", () => {
    expect(buildOrgCreateBody({ ...emptyOrgFormValues, organization_alias: "acme" })).toStrictEqual({
      organization_alias: "acme",
      models: [],
    });
  });

  it("maps every field when the whole form is filled", () => {
    const filledForm = {
      organization_alias: "acme",
      models: ["gpt-5.2"],
      max_budget: "12.5",
      budget_duration: "30d",
      tpm_limit: "1000",
      rpm_limit: "50",
      vector_stores: ["vs-1"],
      mcp: { servers: ["srv-1"], accessGroups: ["ag-1"], toolsets: ["ts-1"] },
      metadata: '{"env": "prod"}',
    };
    const expectedBody = {
      organization_alias: "acme",
      models: ["gpt-5.2"],
      max_budget: 12.5,
      budget_duration: "30d",
      tpm_limit: 1000,
      rpm_limit: 50,
      metadata: { env: "prod" },
      object_permission: {
        vector_stores: ["vs-1"],
        mcp_servers: ["srv-1"],
        mcp_access_groups: ["ag-1"],
        mcp_toolsets: ["ts-1"],
      },
    };

    expect(buildOrgCreateBody(filledForm)).toStrictEqual(expectedBody);
  });

  it("includes only the non-empty grant lists in object_permission", () => {
    expect(
      buildOrgCreateBody({
        ...emptyOrgFormValues,
        organization_alias: "acme",
        mcp: { servers: [], accessGroups: [], toolsets: ["ts-1"] },
      }).object_permission,
    ).toStrictEqual({ mcp_toolsets: ["ts-1"] });
  });

  it("parses metadata into an object instead of sending the raw string", () => {
    expect(
      buildOrgCreateBody({ ...emptyOrgFormValues, organization_alias: "acme", metadata: '{"a": 1}' }).metadata,
    ).toStrictEqual({ a: 1 });
  });
});
