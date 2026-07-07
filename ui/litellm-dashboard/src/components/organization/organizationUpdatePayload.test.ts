import { describe, expect, it } from "vitest";

import {
  buildOrganizationUpdateV2Payload,
  OrgMetadataParseError,
  parseMetadata,
  toNumberOrNull,
  type OrgSettingsFormValues,
} from "./organizationUpdatePayload";

const touched =
  (...names: string[]) =>
  (name: string): boolean =>
    names.includes(name);

describe("toNumberOrNull", () => {
  it("maps empty-ish and non-numeric inputs to null", () => {
    expect(toNumberOrNull("")).toBeNull();
    expect(toNumberOrNull(null)).toBeNull();
    expect(toNumberOrNull(undefined)).toBeNull();
    expect(toNumberOrNull("not-a-number")).toBeNull();
  });

  it("coerces numeric strings and numbers (including 0)", () => {
    expect(toNumberOrNull("500")).toBe(500);
    expect(toNumberOrNull(0)).toBe(0);
    expect(toNumberOrNull(12.5)).toBe(12.5);
  });
});

describe("parseMetadata", () => {
  it("treats empty / whitespace / undefined as null", () => {
    expect(parseMetadata("")).toBeNull();
    expect(parseMetadata("   ")).toBeNull();
    expect(parseMetadata(undefined)).toBeNull();
  });

  it("parses a JSON object", () => {
    expect(parseMetadata('{"a":1}')).toEqual({ a: 1 });
  });

  it("treats literal null as cleared", () => {
    expect(parseMetadata("null")).toBeNull();
  });

  it("rejects arrays, primitives, and invalid JSON", () => {
    expect(() => parseMetadata("[1,2]")).toThrow(OrgMetadataParseError);
    expect(() => parseMetadata('"a string"')).toThrow(OrgMetadataParseError);
    expect(() => parseMetadata("{not json")).toThrow(OrgMetadataParseError);
  });
});

describe("buildOrganizationUpdateV2Payload", () => {
  it("sends nothing when no field is touched", () => {
    const values: OrgSettingsFormValues = {
      organization_alias: "acme",
      models: ["gpt-4"],
      tpm_limit: 100,
      rpm_limit: 50,
      max_budget: 10,
      budget_duration: "30d",
      metadata: '{"team":"core"}',
      vector_stores: ["vs-1"],
      mcp_servers_and_groups: { servers: ["srv-1"], accessGroups: ["grp-1"] },
    };
    expect(buildOrganizationUpdateV2Payload(values, touched())).toEqual({});
  });

  it("omits a field that has a value but was not touched", () => {
    expect(buildOrganizationUpdateV2Payload({ tpm_limit: 500 }, touched())).toEqual({});
  });

  it("sends a touched alias", () => {
    expect(buildOrganizationUpdateV2Payload({ organization_alias: "new-name" }, touched("organization_alias"))).toEqual(
      {
        organization_alias: "new-name",
      },
    );
  });

  it("sends a touched budget_duration and clears it to null when emptied", () => {
    expect(buildOrganizationUpdateV2Payload({ budget_duration: "7d" }, touched("budget_duration"))).toEqual({
      budget_duration: "7d",
    });
    expect(buildOrganizationUpdateV2Payload({ budget_duration: "" }, touched("budget_duration"))).toEqual({
      budget_duration: null,
    });
  });

  it("sends touched models, including an empty array to clear", () => {
    expect(buildOrganizationUpdateV2Payload({ models: ["a", "b"] }, touched("models"))).toEqual({
      models: ["a", "b"],
    });
    expect(buildOrganizationUpdateV2Payload({ models: [] }, touched("models"))).toEqual({ models: [] });
    expect(buildOrganizationUpdateV2Payload({ models: undefined }, touched("models"))).toEqual({ models: [] });
  });

  it("sends a touched numeric value, including 0", () => {
    expect(buildOrganizationUpdateV2Payload({ tpm_limit: 500 }, touched("tpm_limit"))).toEqual({ tpm_limit: 500 });
    expect(buildOrganizationUpdateV2Payload({ max_budget: 0 }, touched("max_budget"))).toEqual({ max_budget: 0 });
  });

  it("coerces a touched but emptied numeric input to null", () => {
    expect(buildOrganizationUpdateV2Payload({ tpm_limit: "" }, touched("tpm_limit"))).toEqual({ tpm_limit: null });
    expect(buildOrganizationUpdateV2Payload({ rpm_limit: null }, touched("rpm_limit"))).toEqual({ rpm_limit: null });
  });

  it("sends touched metadata as a parsed object", () => {
    expect(buildOrganizationUpdateV2Payload({ metadata: '{"a":1}' }, touched("metadata"))).toEqual({
      metadata: { a: 1 },
    });
  });

  it("clears touched metadata to null when the box is emptied", () => {
    expect(buildOrganizationUpdateV2Payload({ metadata: "" }, touched("metadata"))).toEqual({ metadata: null });
  });

  it("throws on invalid metadata JSON so Save can be blocked", () => {
    expect(() => buildOrganizationUpdateV2Payload({ metadata: "{oops" }, touched("metadata"))).toThrow(
      OrgMetadataParseError,
    );
  });

  it("does not parse untouched metadata, so invalid JSON left untouched never throws", () => {
    expect(buildOrganizationUpdateV2Payload({ metadata: "{oops" }, touched())).toEqual({});
  });

  it("sends object_permission when vector_stores is touched", () => {
    expect(buildOrganizationUpdateV2Payload({ vector_stores: ["vs-1"] }, touched("vector_stores"))).toEqual({
      object_permission: { vector_stores: ["vs-1"], mcp_servers: [], mcp_access_groups: [] },
    });
  });

  it("sends object_permission when mcp_servers_and_groups is touched", () => {
    expect(
      buildOrganizationUpdateV2Payload(
        { mcp_servers_and_groups: { servers: ["srv-1"], accessGroups: ["grp-1"] } },
        touched("mcp_servers_and_groups"),
      ),
    ).toEqual({
      object_permission: { vector_stores: [], mcp_servers: ["srv-1"], mcp_access_groups: ["grp-1"] },
    });
  });

  it("sends the full object_permission with [] clears when either half is touched", () => {
    expect(
      buildOrganizationUpdateV2Payload(
        { vector_stores: ["vs-1"], mcp_servers_and_groups: { servers: [], accessGroups: [] } },
        touched("vector_stores"),
      ),
    ).toEqual({ object_permission: { vector_stores: ["vs-1"], mcp_servers: [], mcp_access_groups: [] } });
  });

  it("omits object_permission when neither vector_stores nor mcp_servers_and_groups is touched", () => {
    expect(
      buildOrganizationUpdateV2Payload(
        { vector_stores: ["vs-1"], mcp_servers_and_groups: { servers: ["srv-1"], accessGroups: [] } },
        touched("organization_alias"),
      ),
    ).toEqual({});
  });

  it("bundles multiple touched sets and clears into one payload", () => {
    const payload = buildOrganizationUpdateV2Payload(
      { tpm_limit: null, rpm_limit: 25, metadata: "" },
      touched("tpm_limit", "rpm_limit", "metadata"),
    );
    expect(payload).toEqual({ tpm_limit: null, rpm_limit: 25, metadata: null });
  });
});
