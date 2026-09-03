import { describe, expect, it } from "vitest";

import { buildDefaultValue, getInitialValueForField, resolveSchemaType } from "./mcpToolSchemaDefaults";

describe("resolveSchemaType", () => {
  it("returns a scalar type unchanged", () => {
    expect(resolveSchemaType("string")).toBe("string");
  });

  it("returns the first non-null entry of a nullable type array", () => {
    expect(resolveSchemaType(["integer", "null"])).toBe("integer");
    expect(resolveSchemaType(["null", "boolean"])).toBe("boolean");
  });

  it("falls back to the first entry when every entry is null", () => {
    expect(resolveSchemaType(["null"])).toBe("null");
  });
});

describe("buildDefaultValue", () => {
  it("defaults a required (non-nullable) integer with no explicit default to 0", () => {
    expect(buildDefaultValue({ type: "integer" })).toBe(0);
  });

  it("defaults a required (non-nullable) boolean with no explicit default to false", () => {
    expect(buildDefaultValue({ type: "boolean" })).toBe(false);
  });

  it("defaults a required (non-nullable) string with no explicit default to an empty string", () => {
    expect(buildDefaultValue({ type: "string" })).toBe("");
  });

  it("leaves a nullable integer with no explicit default undefined, not a synthetic 0", () => {
    expect(buildDefaultValue({ type: ["integer", "null"] })).toBeUndefined();
  });

  it("leaves a nullable boolean with no explicit default undefined, not a synthetic false", () => {
    expect(buildDefaultValue({ type: ["boolean", "null"] })).toBeUndefined();
  });

  it("defaults a nullable string with no explicit default to an empty string, unaffected by nullability", () => {
    expect(buildDefaultValue({ type: ["string", "null"] })).toBe("");
  });

  it("still honors an explicit default on a nullable numeric field", () => {
    expect(buildDefaultValue({ type: ["integer", "null"], default: 7 })).toBe(7);
  });
});

describe("getInitialValueForField", () => {
  it("passes through undefined for a nullable numeric field with no default", () => {
    expect(getInitialValueForField({ type: ["integer", "null"] })).toBeUndefined();
  });

  it("JSON-stringifies an object field's computed default", () => {
    expect(getInitialValueForField({ type: "object", properties: { a: { type: "string" } } })).toBe(
      JSON.stringify({ a: "" }, null, 2),
    );
  });
});
