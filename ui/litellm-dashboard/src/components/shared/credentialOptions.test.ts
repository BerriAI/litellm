import { describe, expect, it } from "vitest";

import type { CredentialItem } from "@/components/networking";

import { credentialOptions, NO_CREDENTIAL_OPTION, toCredentialOption } from "./credentialOptions";

describe("credentialOptions", () => {
  it("leads with the None option whose value is the empty string", () => {
    const options = credentialOptions([]);
    expect(options[0]).toEqual({ label: "None", value: "" });
    expect(NO_CREDENTIAL_OPTION.value).toBe("");
  });

  it("keeps the credential name as the option value and shows the alias as the sublabel", () => {
    const credential: CredentialItem = {
      credential_name: "openai-main",
      credential_alias: "Prod OpenAI",
      credential_values: {},
      credential_info: {},
    };

    expect(toCredentialOption(credential)).toEqual({
      label: "openai-main",
      value: "openai-main",
      sublabel: "Prod OpenAI",
    });
  });

  it.each([null, undefined])("leaves the sublabel undefined when the alias is %s", (alias) => {
    const credential: CredentialItem = {
      credential_name: "plain",
      credential_alias: alias,
      credential_values: {},
      credential_info: {},
    };

    expect(toCredentialOption(credential).sublabel).toBeUndefined();
  });

  it("maps every credential after the None option", () => {
    const credentials: CredentialItem[] = [
      { credential_name: "a", credential_values: {}, credential_info: {} },
      { credential_name: "b", credential_alias: "Bee", credential_values: {}, credential_info: {} },
    ];

    const options = credentialOptions(credentials);
    expect(options.map((option) => option.value)).toEqual(["", "a", "b"]);
    expect(options[2].sublabel).toBe("Bee");
  });
});
