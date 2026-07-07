import { describe, expect, it } from "vitest";

import {
  buildOrganizationUpdateV2Payload,
  buildOrgSettingsBaseline,
  OrgMetadataParseError,
  parseMetadata,
  toNumberOrNull,
  type OrgSettingsBaseline,
} from "./organizationUpdatePayload";

const baseline = (overrides: Partial<OrgSettingsBaseline> = {}): OrgSettingsBaseline => ({
  organization_alias: "acme",
  models: ["gpt-4"],
  tpm_limit: 100,
  rpm_limit: 50,
  max_budget: 10,
  budget_duration: "30d",
  metadata: { team: "core" },
  ...overrides,
});

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
  it("sends a changed limit", () => {
    expect(buildOrganizationUpdateV2Payload({ values: { tpm_limit: 500 }, baseline: baseline() })).toEqual({
      tpm_limit: 500,
    });
  });

  it("sends null for a cleared limit", () => {
    expect(buildOrganizationUpdateV2Payload({ values: { tpm_limit: null }, baseline: baseline() })).toEqual({
      tpm_limit: null,
    });
  });

  it("coerces an emptied numeric input to null, never an empty string", () => {
    expect(buildOrganizationUpdateV2Payload({ values: { tpm_limit: "" }, baseline: baseline() })).toEqual({
      tpm_limit: null,
    });
  });

  it("omits a limit whose value is unchanged", () => {
    expect(buildOrganizationUpdateV2Payload({ values: { tpm_limit: 100 }, baseline: baseline() })).toEqual({});
  });

  it("omits an absent (untouched) limit", () => {
    expect(buildOrganizationUpdateV2Payload({ values: {}, baseline: baseline() })).toEqual({});
  });

  it("sets metadata", () => {
    expect(
      buildOrganizationUpdateV2Payload({ values: { metadata: '{"a":1}' }, baseline: baseline({ metadata: null }) }),
    ).toEqual({ metadata: { a: 1 } });
  });

  it("edits a metadata value", () => {
    expect(
      buildOrganizationUpdateV2Payload({ values: { metadata: '{"a":2}' }, baseline: baseline({ metadata: { a: 1 } }) }),
    ).toEqual({ metadata: { a: 2 } });
  });

  it("clears metadata to null when the box is emptied", () => {
    expect(
      buildOrganizationUpdateV2Payload({ values: { metadata: "" }, baseline: baseline({ metadata: { a: 1 } }) }),
    ).toEqual({ metadata: null });
  });

  it("omits metadata that is deep-equal to the baseline", () => {
    expect(buildOrganizationUpdateV2Payload({ values: { metadata: '{"team":"core"}' }, baseline: baseline() })).toEqual(
      {},
    );
  });

  it("throws on invalid metadata JSON so Save can be blocked", () => {
    expect(() => buildOrganizationUpdateV2Payload({ values: { metadata: "{oops" }, baseline: baseline() })).toThrow(
      OrgMetadataParseError,
    );
  });

  it("sends changed and cleared models", () => {
    expect(
      buildOrganizationUpdateV2Payload({ values: { models: ["a", "b"] }, baseline: baseline({ models: ["a"] }) }),
    ).toEqual({ models: ["a", "b"] });
    expect(buildOrganizationUpdateV2Payload({ values: { models: [] }, baseline: baseline({ models: ["a"] }) })).toEqual(
      { models: [] },
    );
  });

  it("omits models that are unchanged", () => {
    expect(buildOrganizationUpdateV2Payload({ values: { models: ["gpt-4"] }, baseline: baseline() })).toEqual({});
  });

  it("sends a changed alias and budget_duration", () => {
    expect(
      buildOrganizationUpdateV2Payload({ values: { organization_alias: "new-name" }, baseline: baseline() }),
    ).toEqual({ organization_alias: "new-name" });
    expect(buildOrganizationUpdateV2Payload({ values: { budget_duration: "7d" }, baseline: baseline() })).toEqual({
      budget_duration: "7d",
    });
  });

  it("returns an empty body when a full form submit changed nothing", () => {
    const values = {
      organization_alias: "acme",
      models: ["gpt-4"],
      tpm_limit: 100,
      rpm_limit: 50,
      max_budget: 10,
      budget_duration: "30d",
      metadata: '{"team":"core"}',
    };
    expect(buildOrganizationUpdateV2Payload({ values, baseline: baseline() })).toEqual({});
  });

  it("bundles multiple sets and clears into one payload", () => {
    const payload = buildOrganizationUpdateV2Payload({
      values: { tpm_limit: null, rpm_limit: 25, metadata: "" },
      baseline: baseline(),
    });
    expect(payload).toEqual({ tpm_limit: null, rpm_limit: 25, metadata: null });
  });
});

describe("buildOrgSettingsBaseline", () => {
  it("extracts a baseline from an org row", () => {
    expect(
      buildOrgSettingsBaseline({
        organization_alias: "acme",
        models: ["gpt-4"],
        metadata: { team: "core" },
        litellm_budget_table: { tpm_limit: 100, rpm_limit: 50, max_budget: 10, budget_duration: "30d" },
      }),
    ).toEqual({
      organization_alias: "acme",
      models: ["gpt-4"],
      tpm_limit: 100,
      rpm_limit: 50,
      max_budget: 10,
      budget_duration: "30d",
      metadata: { team: "core" },
    });
  });

  it("defaults a missing budget table and metadata to null", () => {
    const result = buildOrgSettingsBaseline({});
    expect(result.tpm_limit).toBeNull();
    expect(result.budget_duration).toBeNull();
    expect(result.metadata).toBeNull();
    expect(result.models).toBeNull();
  });
});
