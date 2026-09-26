import { describe, expect, it } from "vitest";
import type { KeyResponse } from "../key_team_helpers/key_list";
import { keyEditFormSchema, toKeyEditFormValues, toSubmittedValues } from "./keyEditFormValues";

const parse = (values: Record<string, unknown>) => keyEditFormSchema.safeParse(values);

describe("tpd_limit round trip", () => {
  const keyData = { token: "tok", models: [], rpm_limit: 5, tpd_limit: 250000 } as unknown as KeyResponse;

  it("hydrates the stored daily batch budget into the edit form", () => {
    expect(toKeyEditFormValues(keyData)).toMatchObject({ rpm_limit: 5, tpd_limit: 250000 });
  });

  it("submits tpd_limit next to the minute limits", () => {
    const submitted = toSubmittedValues(toKeyEditFormValues(keyData), { canViewPolicies: true, canViewPrompts: true });
    expect(submitted).toMatchObject({ rpm_limit: 5, tpd_limit: 250000 });
  });

  it("submits null when the operator cleared tpd_limit", () => {
    const submitted = toSubmittedValues(
      { ...toKeyEditFormValues(keyData), tpd_limit: null },
      { canViewPolicies: true, canViewPrompts: true },
    );
    expect(submitted.tpd_limit).toBeNull();
  });
});

describe("workload_class round trip", () => {
  const gates = { canViewPolicies: true, canViewPrompts: true };
  const keyData = {
    token: "tok",
    models: [],
    metadata: { priority: "batch", region: "us" },
  } as unknown as KeyResponse;

  it("lifts metadata.priority into the selector and hides it from the free-form metadata JSON", () => {
    const values = toKeyEditFormValues(keyData);
    expect(values.workload_class).toBe("batch");
    expect(JSON.parse(values.metadata ?? "{}")).toEqual({ region: "us" });
  });

  it("writes the newly picked class back into metadata.priority next to untouched keys", () => {
    const submitted = toSubmittedValues({ ...toKeyEditFormValues(keyData), workload_class: "production" }, gates);
    expect(JSON.parse(String(submitted.metadata))).toEqual({ region: "us", priority: "production" });
  });

  it("drops metadata.priority when the key moves back to the default pool", () => {
    const submitted = toSubmittedValues({ ...toKeyEditFormValues(keyData), workload_class: "default" }, gates);
    expect(JSON.parse(String(submitted.metadata))).toEqual({ region: "us" });
  });
});

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
