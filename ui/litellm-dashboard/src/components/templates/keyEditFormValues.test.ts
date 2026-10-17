import { describe, expect, it } from "vitest";
import type { KeyResponse } from "../key_team_helpers/key_list";
import { keyEditFormSchema, toKeyEditFormValues, toSubmittedValues } from "./keyEditFormValues";

const parse = (values: Record<string, unknown>) => keyEditFormSchema.safeParse(values);

describe("key model aliases", () => {
  it("loads key aliases independently of team aliases", () => {
    const keyData = {
      aliases: { chat: "chat-dev" },
      team_model_aliases: { chat: "team-chat" },
    } as unknown as KeyResponse;
    expect(toKeyEditFormValues(keyData).aliases).toEqual(keyData.aliases);
  });

  it("preserves aliases through schema validation", () => {
    expect(keyEditFormSchema.parse({ aliases: { chat: "chat-dev" } })).toEqual({ aliases: { chat: "chat-dev" } });
  });

  it("rejects alias targets that are not model names", () => {
    expect(parse({ aliases: { chat: 123 } }).success).toBe(false);
  });

  it("submits a changed mapping including an explicitly emptied one", () => {
    const gates = { canViewPolicies: false, canViewPrompts: false, aliasesChanged: true };
    expect(toSubmittedValues({ aliases: { chat: "chat-prod" } }, gates).aliases).toEqual({ chat: "chat-prod" });
    expect(toSubmittedValues({ aliases: {} }, gates).aliases).toEqual({});
  });

  it("omits aliases when the field was not changed", () => {
    expect(
      toSubmittedValues({ aliases: { chat: "chat-dev" } }, { canViewPolicies: false, canViewPrompts: false }),
    ).not.toHaveProperty("aliases");
  });
});

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
