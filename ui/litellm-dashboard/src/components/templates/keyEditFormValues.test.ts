import { describe, expect, it } from "vitest";
import { keyEditFormSchema } from "./keyEditFormValues";

const parse = (values: Record<string, unknown>) => keyEditFormSchema.safeParse(values);

describe("keyEditFormSchema", () => {
  it("accepts an empty form", () => {
    expect(parse({}).success).toBe(true);
  });

  it("rejects a fractional estimated output tokens value", () => {
    expect(parse({ default_estimated_output_tokens: "12.5" }).success).toBe(false);
  });

  it("rejects a zero or negative estimated output tokens value", () => {
    expect(parse({ default_estimated_output_tokens: "-5" }).success).toBe(false);
    expect(parse({ default_estimated_output_tokens: 0 }).success).toBe(false);
  });

  it("accepts a blank or absent estimated output tokens value", () => {
    expect(parse({ default_estimated_output_tokens: "" }).success).toBe(true);
    expect(parse({ default_estimated_output_tokens: null }).success).toBe(true);
  });

  it("rejects a per-model estimate that is not a JSON object of positive integers", () => {
    expect(parse({ default_estimated_output_tokens_per_model: "not json" }).success).toBe(false);
    expect(parse({ default_estimated_output_tokens_per_model: '{"gpt-4": 0}' }).success).toBe(false);
  });

  it("accepts a per-model estimate that is a JSON object of positive integers", () => {
    expect(parse({ default_estimated_output_tokens_per_model: '{"gpt-4": 4096}' }).success).toBe(true);
  });
});
