import { describe, expect, it } from "vitest";

import type { CredentialItem } from "@/components/networking";

import {
  credentialLabel,
  credentialLabelsByName,
  credentialOptions,
  NO_CREDENTIAL_OPTION,
  toCredentialOption,
} from "./credentialOptions";

const credential = (credential_name: string, display_name?: string | null): CredentialItem => ({
  credential_name,
  display_name,
  credential_values: {},
  credential_info: {},
});

describe("credentialOptions", () => {
  it("leads with the None option whose value is the empty string", () => {
    const options = credentialOptions([]);
    expect(options[0]).toEqual({ label: "None", value: "" });
    expect(NO_CREDENTIAL_OPTION.value).toBe("");
  });

  it("shows the display name as the label and keeps the credential name as value and sublabel", () => {
    expect(toCredentialOption(credential("openai-main", "Prod OpenAI"))).toEqual({
      label: "Prod OpenAI",
      value: "openai-main",
      sublabel: "openai-main",
    });
  });

  it.each([null, undefined, ""])("falls back to the credential name when the display name is %s", (displayName) => {
    expect(toCredentialOption(credential("plain", displayName))).toEqual({
      label: "plain",
      value: "plain",
      sublabel: undefined,
    });
    expect(credentialLabel(credential("plain", displayName))).toBe("plain");
  });

  it("maps every credential after the None option", () => {
    const options = credentialOptions([credential("a"), credential("b", "Bee")]);
    expect(options.map((option) => option.value)).toEqual(["", "a", "b"]);
    expect(options.map((option) => option.label)).toEqual(["None", "a", "Bee"]);
  });

  it("indexes labels by credential name", () => {
    const labels = credentialLabelsByName([credential("a"), credential("b", "Bee")]);
    expect([...labels.entries()]).toEqual([
      ["a", "a"],
      ["b", "Bee"],
    ]);
  });
});
