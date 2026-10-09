import { describe, expect, it } from "vitest";
import {
  identitySourceOptions,
  inferIdentitySource,
  isAnthropicProvider,
  MAX_ISSUER_TTL_SECONDS,
  validateFederationValueStored,
  validateIdentityTokenReference,
  validateIssuerTtlSeconds,
} from "./anthropic_federation";

const storedKeycloak = {
  anthropic_federation_rule_id: "fdrl_stored",
  anthropic_organization_id: "org-stored",
  anthropic_identity_source: "keycloak",
  anthropic_keycloak_token_url: "http****",
  anthropic_keycloak_client_id: "lite****",
  anthropic_keycloak_client_secret_ref: "os.e****",
  anthropic_keycloak_auth_method: "clie****",
  anthropic_keycloak_scope: "open****",
};

describe("isAnthropicProvider", () => {
  it.each(["Anthropic", "anthropic"])("accepts %s", (provider) => {
    expect(isAnthropicProvider(provider)).toBe(true);
  });

  it.each(["OpenAI", "ANTHROPIC_TEXT", "Anthropic Text", null, undefined])("rejects %s", (provider) => {
    expect(isAnthropicProvider(provider)).toBe(false);
  });
});

describe("reading a stored identity source", () => {
  it("lets a declared identity source win over a leftover token file", () => {
    expect(
      inferIdentitySource({ anthropic_identity_source: "internal_issuer", anthropic_identity_token_file: "/var****" }),
    ).toBe("internal_issuer");
    expect(inferIdentitySource(storedKeycloak)).toBe("keycloak");
  });

  it("prefers the token file over the secret reference, the order the proxy resolves them in", () => {
    expect(
      inferIdentitySource({ anthropic_identity_token_file: "/var****", anthropic_identity_token: "oidc****" }),
    ).toBe("token_file");
    expect(inferIdentitySource({ anthropic_identity_token: "oidc****" })).toBe("secret_reference");
  });

  it("falls back to the proxy environment when no identity value is stored", () => {
    expect(inferIdentitySource({ anthropic_federation_rule_id: "fdrl_1" })).toBe("environment");
  });

  it("keeps a stored identity source it does not offer as its own option, never as another source", () => {
    expect(inferIdentitySource({ anthropic_identity_source: "spiffe" })).toBe("unrecognized");
    expect(identitySourceOptions({ anthropic_identity_source: "spiffe" })[0]).toEqual({
      value: "unrecognized",
      label: "Stored: spiffe",
    });
    expect(identitySourceOptions(storedKeycloak).map((option) => option.value)).not.toContain("unrecognized");
  });
});

describe("field validation", () => {
  it.each(["oidc/env/ANTHROPIC_IDENTITY_TOKEN", "oidc/file//var/run/secrets/token", "oidc/github/api.anthropic.com"])(
    "accepts the secret reference %s",
    (reference) => {
      expect(validateIdentityTokenReference(reference)).toBe(true);
    },
  );

  it.each(["eyJhbGciOiJSUzI1NiJ9.payload.signature", "oidc/env_path/TOKEN_PATH", "os.environ/TOKEN"])(
    "refuses %s, which the proxy would reject on every request",
    (reference) => {
      expect(validateIdentityTokenReference(reference)).toEqual(expect.stringContaining("oidc/"));
    },
  );

  it("leaves an untouched masked reference and an empty one to the other rules", () => {
    expect(validateIdentityTokenReference("oidc****")).toBe(true);
    expect(validateIdentityTokenReference("")).toBe(true);
  });

  it.each(["1", "300", String(MAX_ISSUER_TTL_SECONDS), 300, ""])("accepts the lifetime %s", (ttl) => {
    expect(validateIssuerTtlSeconds(ttl)).toBe(true);
  });

  it.each(["0", "-5", "1.5", "abc", String(MAX_ISSUER_TTL_SECONDS + 1)])("refuses the lifetime %s", (ttl) => {
    expect(validateIssuerTtlSeconds(ttl)).toEqual(expect.stringContaining("whole number"));
  });

  it("refuses the proxy environment source when every id is blank, since the proxy rejects a credential with no values", () => {
    const rule = validateFederationValueStored("environment");
    const blankIds = {
      api_base: "https://gateway.example.com",
      anthropic_federation_rule_id: "",
      anthropic_organization_id: " ",
      anthropic_service_account_id: undefined,
    };
    expect(rule("", blankIds)).toEqual(expect.stringContaining("at least one"));
    expect(rule("", { ...blankIds, anthropic_federation_workspace_id: "wrkspc_1" })).toBe(true);
    expect(validateFederationValueStored("token_file")("", blankIds)).toBe(true);
  });
});
