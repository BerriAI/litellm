import { describe, expect, it } from "vitest";

import { computeRemovedLitellmParamKeys, isMaskedSecret, stripMaskedSecrets } from "./maskedSecretUtils";

describe("isMaskedSecret", () => {
  it("flags values with a run of 2+ mask chars", () => {
    expect(isMaskedSecret("sk-1****2345")).toBe(true);
  });

  it("does not flag a single star (real config like a wildcard model name)", () => {
    expect(isMaskedSecret("openai/*")).toBe(false);
  });

  it("only flags strings", () => {
    expect(isMaskedSecret(5)).toBe(false);
    expect(isMaskedSecret(null)).toBe(false);
  });
});

describe("computeRemovedLitellmParamKeys", () => {
  it("reports a key the user deleted from the editor", () => {
    const stored = { model: "gpt-5.6-sol", reasoning_effort: "none" };
    const saved = { model: "gpt-5.6-sol" };
    expect(computeRemovedLitellmParamKeys(stored, saved)).toEqual(["reasoning_effort"]);
  });

  it("reports nothing when every stored key is still present", () => {
    const stored = { model: "gpt-5.6-sol", reasoning_effort: "none" };
    const saved = { model: "gpt-5.6-sol", reasoning_effort: "high" };
    expect(computeRemovedLitellmParamKeys(stored, saved)).toEqual([]);
  });

  it("treats a key present with an undefined/null value as still there, not removed", () => {
    // Dedicated form fields spread into the saved blob keep the key even when empty,
    // so an emptied field must not be misread as a deletion.
    const stored = { model: "gpt-5.6-sol", api_base: "https://old.example" };
    const saved = { model: "gpt-5.6-sol", api_base: undefined };
    expect(computeRemovedLitellmParamKeys(stored, saved)).toEqual([]);
  });

  it("never reports a masked secret, so a stripped credential is not nulled out", () => {
    const stored = { model: "gpt-5.6-sol", api_key: "sk-1****2345" };
    const saved = { model: "gpt-5.6-sol" };
    expect(computeRemovedLitellmParamKeys(stored, saved)).toEqual([]);
  });

  it("never reports litellm_credential_name, which has its own save path", () => {
    const stored = { model: "gpt-5.6-sol", litellm_credential_name: "openai-cred" };
    const saved = { model: "gpt-5.6-sol" };
    expect(computeRemovedLitellmParamKeys(stored, saved)).toEqual([]);
  });

  it("reports only the removed non-secret keys among several changes", () => {
    const stored = {
      model: "gpt-5.6-sol",
      reasoning_effort: "none",
      temperature: 0.5,
      api_key: "sk-1****2345",
    };
    const saved = { model: "gpt-5.6-sol", temperature: 0.5 };
    expect(computeRemovedLitellmParamKeys(stored, saved).sort()).toEqual(["reasoning_effort"]);
  });
});

describe("stripMaskedSecrets", () => {
  it("drops keys whose value is a masked secret and keeps the rest", () => {
    expect(stripMaskedSecrets({ model: "gpt-5.6-sol", api_key: "sk-1****2345", reasoning_effort: null })).toEqual({
      model: "gpt-5.6-sol",
      reasoning_effort: null,
    });
  });
});
