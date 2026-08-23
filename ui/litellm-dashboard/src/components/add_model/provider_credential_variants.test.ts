import { describe, expect, it } from "vitest";
import { getVariant, inferActiveVariant, resolveVariantFieldDefs } from "./provider_credential_variants";
import type { ProviderCredentialVariants } from "../networking";

const field = (key: string, required = false): ProviderCredentialVariants["field_definitions"][number] => ({
  key,
  label: key,
  required,
  field_type: "text",
});

const anthropicVariants: ProviderCredentialVariants = {
  selector_label: "Authentication method",
  default_variant: "api_key",
  field_definitions: [
    field("api_base"),
    field("api_key"),
    field("anthropic_federation_rule_id", true),
    field("anthropic_organization_id", true),
    field("anthropic_identity_token", true),
    field("anthropic_identity_token_file", true),
    field("anthropic_issuer_url", true),
    field("anthropic_issuer_signing_key_ref", true),
    field("anthropic_keycloak_token_url", true),
    field("anthropic_keycloak_client_id", true),
    field("anthropic_keycloak_client_secret_ref", true),
  ],
  variants: [
    { id: "api_key", label: "API Key", field_keys: ["api_base", "api_key"], fixed_values: {} },
    {
      id: "wif_token",
      label: "WIF (token)",
      field_keys: ["anthropic_federation_rule_id", "anthropic_organization_id", "anthropic_identity_token"],
      fixed_values: {},
    },
    {
      id: "wif_token_file",
      label: "WIF (token file)",
      field_keys: ["anthropic_federation_rule_id", "anthropic_organization_id", "anthropic_identity_token_file"],
      fixed_values: {},
    },
    {
      id: "wif_internal_issuer",
      label: "WIF (internal issuer)",
      field_keys: ["anthropic_federation_rule_id", "anthropic_organization_id", "anthropic_issuer_url", "anthropic_issuer_signing_key_ref"],
      fixed_values: { anthropic_identity_source: "internal_issuer" },
    },
    {
      id: "wif_keycloak",
      label: "WIF (keycloak)",
      field_keys: [
        "anthropic_federation_rule_id",
        "anthropic_organization_id",
        "anthropic_keycloak_token_url",
        "anthropic_keycloak_client_id",
        "anthropic_keycloak_client_secret_ref",
      ],
      fixed_values: { anthropic_identity_source: "keycloak" },
    },
  ],
};

describe("getVariant", () => {
  it("finds a variant by id", () => {
    expect(getVariant(anthropicVariants, "wif_keycloak")?.label).toBe("WIF (keycloak)");
  });

  it("returns undefined for an unknown id", () => {
    expect(getVariant(anthropicVariants, "nope")).toBeUndefined();
  });
});

describe("resolveVariantFieldDefs", () => {
  it("resolves field_keys to their field_definitions, in order", () => {
    const fields = resolveVariantFieldDefs(anthropicVariants, "wif_token_file");
    expect(fields.map((f) => f.key)).toEqual([
      "anthropic_federation_rule_id",
      "anthropic_organization_id",
      "anthropic_identity_token_file",
    ]);
  });

  it("returns an empty list for an unknown variant id", () => {
    expect(resolveVariantFieldDefs(anthropicVariants, "nope")).toEqual([]);
  });

  it("drops a field_key that has no matching field_definitions entry, rather than crashing", () => {
    const variants: ProviderCredentialVariants = {
      ...anthropicVariants,
      variants: [{ id: "broken", label: "Broken", field_keys: ["api_key", "missing_field"], fixed_values: {} }],
    };
    expect(resolveVariantFieldDefs(variants, "broken").map((f) => f.key)).toEqual(["api_key"]);
  });
});

describe("inferActiveVariant", () => {
  it("defaults to default_variant on a blank form", () => {
    expect(inferActiveVariant(anthropicVariants, {})).toBe("api_key");
  });

  it("picks wif_token when only the inline identity token is set", () => {
    const values = {
      anthropic_federation_rule_id: "rule-1",
      anthropic_organization_id: "org-1",
      anthropic_identity_token: "oidc/env/TOKEN",
    };
    expect(inferActiveVariant(anthropicVariants, values)).toBe("wif_token");
  });

  it("picks wif_token_file over wif_token when the file field is set instead", () => {
    const values = {
      anthropic_federation_rule_id: "rule-1",
      anthropic_organization_id: "org-1",
      anthropic_identity_token_file: "/var/run/secrets/tokens/oidc-token",
    };
    expect(inferActiveVariant(anthropicVariants, values)).toBe("wif_token_file");
  });

  it("picks wif_internal_issuer from its fixed discriminator, not just field presence", () => {
    const values = {
      anthropic_identity_source: "internal_issuer",
      anthropic_federation_rule_id: "rule-1",
      anthropic_organization_id: "org-1",
      anthropic_issuer_url: "https://issuer.example.com",
      anthropic_issuer_signing_key_ref: "os.environ/SIGNING_KEY",
    };
    expect(inferActiveVariant(anthropicVariants, values)).toBe("wif_internal_issuer");
  });

  it("picks wif_keycloak from its fixed discriminator", () => {
    const values = {
      anthropic_identity_source: "keycloak",
      anthropic_federation_rule_id: "rule-1",
      anthropic_organization_id: "org-1",
      anthropic_keycloak_token_url: "https://keycloak.example.com/token",
      anthropic_keycloak_client_id: "client-1",
      anthropic_keycloak_client_secret_ref: "os.environ/SECRET",
    };
    expect(inferActiveVariant(anthropicVariants, values)).toBe("wif_keycloak");
  });

  it("does not match a fixed-values variant whose required fields are still incomplete", () => {
    // The discriminator alone (e.g. left over from a variant switch) is not enough.
    const values = { anthropic_identity_source: "keycloak" };
    expect(inferActiveVariant(anthropicVariants, values)).toBe("api_key");
  });

  it("ignores non-required optional fields when deciding a match", () => {
    const values = { api_base: "https://api.anthropic.com" };
    expect(inferActiveVariant(anthropicVariants, values)).toBe("api_key");
  });
});
