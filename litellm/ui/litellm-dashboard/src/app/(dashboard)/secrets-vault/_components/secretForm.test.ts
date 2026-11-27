import { describe, expect, it } from "vitest";

import { secretFormSchema, toSecretWrite } from "./secretForm";

const valid = { name: "GITHUB_TOKEN", value: "ghp_example", description: "" };

describe("secretFormSchema", () => {
  it.each(["GITHUB_TOKEN", "npm.token-2", "a", "x".repeat(128)])("accepts the name %s", (name) => {
    expect(secretFormSchema.safeParse({ ...valid, name }).success).toBe(true);
  });

  it.each(["", "has space", "-leading-dash", "path/segment", "x".repeat(129)])("rejects the name %j", (name) => {
    expect(secretFormSchema.safeParse({ ...valid, name }).success).toBe(false);
  });

  it("rejects an empty value", () => {
    expect(secretFormSchema.safeParse({ ...valid, value: "" }).success).toBe(false);
  });
});

describe("toSecretWrite", () => {
  it("sends the value untouched and a blank description as null", () => {
    expect(toSecretWrite({ ...valid, value: "  padded value\n", description: "   " })).toEqual({
      value: "  padded value\n",
      description: null,
    });
  });

  it("trims a description that has text", () => {
    expect(toSecretWrite({ ...valid, description: "  CI token " }).description).toBe("CI token");
  });
});
