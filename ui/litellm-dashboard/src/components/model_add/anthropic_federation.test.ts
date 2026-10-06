import { describe, expect, it } from "vitest";
import {
  buildCreateCredentialValues,
  buildCredentialPatch,
  buildProviderChangePatch,
  inferAuthMethod,
  identitySourceOptions,
  inferIdentitySource,
  isAnthropicProvider,
  isFederatedCredential,
  MAX_ISSUER_TTL_SECONDS,
  validateFederationValueStored,
  validateIdentityTokenReference,
  validateIssuerTtlSeconds,
  validateMaskedValueUntouched,
} from "./anthropic_federation";

const apiKeySelection = { authMethod: "api_key", identitySource: "token_file" } as const;
const tokenFileSelection = { authMethod: "federation", identitySource: "token_file" } as const;
const internalIssuerSelection = { authMethod: "federation", identitySource: "internal_issuer" } as const;
const keycloakSelection = { authMethod: "federation", identitySource: "keycloak" } as const;
const environmentSelection = { authMethod: "federation", identitySource: "environment" } as const;

const storedTokenFile = {
  api_base: "https://api.anthropic.com",
  anthropic_federation_rule_id: "fdrl_stored",
  anthropic_organization_id: "org-stored",
  anthropic_federation_workspace_id: "wrkspc_stored",
  anthropic_identity_token_file: "/var****",
};

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

describe("reading a stored credential", () => {
  it("treats a credential with any federation value as federated", () => {
    expect(isFederatedCredential({ anthropic_federation_rule_id: "fdrl_1" })).toBe(true);
    expect(inferAuthMethod({ anthropic_identity_source: "keycloak" })).toBe("federation");
  });

  it("treats an api key credential, an empty one, and a missing one as not federated", () => {
    expect(isFederatedCredential({ api_key: "sk-1****", api_base: "https://api.anthropic.com" })).toBe(false);
    expect(isFederatedCredential({ anthropic_federation_rule_id: "" })).toBe(false);
    expect(inferAuthMethod(undefined)).toBe("api_key");
  });

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

  it("opens a credential that stores an api key next to federation values as an api key credential", () => {
    expect(inferAuthMethod({ api_key: "sk-1****", anthropic_federation_rule_id: "fdrl_1" })).toBe("api_key");
    expect(inferAuthMethod({ api_key: "", anthropic_federation_rule_id: "fdrl_1" })).toBe("federation");
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

  it("refuses a hidden stored value that was only partly edited", () => {
    const rule = validateMaskedValueUntouched("os.e****");
    expect(rule("os.e****")).toBe(true);
    expect(rule("os.environ/NEW_REF")).toBe(true);
    expect(rule("os.e****_NEW")).toEqual(expect.stringContaining("Replace the whole value"));
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

describe("buildCreateCredentialValues", () => {
  it("sends the typed federation values and nothing for the fields left empty", () => {
    const typedValues = {
      api_base: "",
      anthropic_federation_rule_id: " fdrl_new\n",
      anthropic_organization_id: "org-new",
      anthropic_service_account_id: "",
      anthropic_federation_workspace_id: undefined,
      anthropic_identity_token_file: "/var/run/secrets/anthropic/token",
    };
    expect(buildCreateCredentialValues(typedValues, tokenFileSelection)).toEqual({
      anthropic_federation_rule_id: "fdrl_new",
      anthropic_organization_id: "org-new",
      anthropic_identity_token_file: "/var/run/secrets/anthropic/token",
    });
  });

  it("names the identity source and sends the lifetime as a number for the internal issuer", () => {
    const typedValues = {
      anthropic_federation_rule_id: "fdrl_new",
      anthropic_organization_id: "org-new",
      anthropic_issuer_url: "https://litellm.example.com",
      anthropic_issuer_subject: "litellm-proxy",
      anthropic_issuer_ttl_seconds: "120",
      anthropic_issuer_signing_key_ref: "os.environ/ISSUER_KEY",
    };
    const values = buildCreateCredentialValues(typedValues, internalIssuerSelection);
    expect(values.anthropic_identity_source).toBe("internal_issuer");
    expect(values.anthropic_issuer_ttl_seconds).toBe(120);
  });

  it("does not name an identity source for an api key credential and keeps its values as typed", () => {
    expect(buildCreateCredentialValues({ api_key: " sk-ant-typed ", api_base: "" }, apiKeySelection)).toEqual({
      api_key: " sk-ant-typed ",
    });
  });
});

describe("buildCredentialPatch", () => {
  it("writes nothing when the admin saves a federated credential untouched", () => {
    expect(
      buildCredentialPatch(storedTokenFile, { ...storedTokenFile }, tokenFileSelection, tokenFileSelection),
    ).toEqual({ credential_values: {}, credential_values_to_delete: [] });
    expect(buildCredentialPatch(storedKeycloak, { ...storedKeycloak }, keycloakSelection, keycloakSelection)).toEqual({
      credential_values: {},
      credential_values_to_delete: [],
    });
  });

  it("sends only the value the admin changed", () => {
    expect(
      buildCredentialPatch(
        storedTokenFile,
        { ...storedTokenFile, anthropic_organization_id: "org-edited" },
        tokenFileSelection,
        tokenFileSelection,
      ),
    ).toEqual({ credential_values: { anthropic_organization_id: "org-edited" }, credential_values_to_delete: [] });
  });

  it("replaces a hidden stored value when the admin types a new one", () => {
    const patch = buildCredentialPatch(
      storedTokenFile,
      { ...storedTokenFile, anthropic_identity_token_file: "/run/secrets/new-token" },
      tokenFileSelection,
      tokenFileSelection,
    );
    expect(patch.credential_values).toEqual({ anthropic_identity_token_file: "/run/secrets/new-token" });
  });

  it("deletes every stored value the admin cleared, a base URL included", () => {
    expect(
      buildCredentialPatch(
        { ...storedTokenFile, api_base: "https://gateway.example.com" },
        { ...storedTokenFile, api_base: "", anthropic_federation_workspace_id: " " },
        tokenFileSelection,
        tokenFileSelection,
      ),
    ).toEqual({
      credential_values: {},
      credential_values_to_delete: ["api_base", "anthropic_federation_workspace_id"],
    });
  });

  it("deletes every stored value the admin did not re-enter when the provider changed", () => {
    const stored = { api_base: "https://corp.openai.azure.com", api_version: "2024-10-21", api_key: "sk-1****" };
    const typed = { anthropic_federation_rule_id: "fdrl_1", anthropic_identity_token_file: "/run/secrets/token" };
    expect(buildProviderChangePatch(stored, typed, tokenFileSelection)).toEqual({
      credential_values: typed,
      credential_values_to_delete: ["api_base", "api_version", "api_key"],
    });
  });

  it("does not treat an unchanged stored lifetime as an edit", () => {
    const stored = {
      ...storedTokenFile,
      anthropic_identity_source: "internal_issuer",
      anthropic_issuer_ttl_seconds: 300,
    };
    const patch = buildCredentialPatch(
      stored,
      { anthropic_issuer_ttl_seconds: "300" },
      internalIssuerSelection,
      internalIssuerSelection,
    );
    expect(patch.credential_values).toEqual({});
  });

  it("drops the other source's stored values when the admin switches identity source", () => {
    const typedValues = {
      anthropic_federation_rule_id: "fdrl_stored",
      anthropic_organization_id: "org-stored",
      anthropic_issuer_url: "https://litellm.example.com",
      anthropic_issuer_subject: "litellm-proxy",
      anthropic_issuer_signing_key_ref: "os.environ/ISSUER_KEY",
    };
    const patch = buildCredentialPatch(storedKeycloak, typedValues, keycloakSelection, internalIssuerSelection);
    const expectedValues = {
      anthropic_identity_source: "internal_issuer",
      anthropic_issuer_url: "https://litellm.example.com",
      anthropic_issuer_subject: "litellm-proxy",
      anthropic_issuer_signing_key_ref: "os.environ/ISSUER_KEY",
    };
    expect(patch.credential_values).toEqual(expectedValues);
    expect([...patch.credential_values_to_delete].sort()).toEqual([
      "anthropic_keycloak_auth_method",
      "anthropic_keycloak_client_id",
      "anthropic_keycloak_client_secret_ref",
      "anthropic_keycloak_scope",
      "anthropic_keycloak_token_url",
    ]);
  });

  it("drops the declared source when the admin switches to a token file or the proxy environment", () => {
    const toEnvironment = buildCredentialPatch(storedKeycloak, {}, keycloakSelection, environmentSelection);
    expect(toEnvironment.credential_values).toEqual({});
    expect(toEnvironment.credential_values_to_delete).toContain("anthropic_identity_source");
    expect(toEnvironment.credential_values_to_delete).not.toContain("anthropic_federation_rule_id");
  });

  it("deletes the stored api key when the admin switches the credential to federation", () => {
    const typedValues = {
      api_base: "https://api.anthropic.com",
      anthropic_federation_rule_id: "fdrl_new",
      anthropic_organization_id: "org-new",
      anthropic_identity_token_file: "/var/run/secrets/anthropic/token",
    };
    const patch = buildCredentialPatch(
      { api_key: "sk-a****", api_base: "https://api.anthropic.com" },
      typedValues,
      apiKeySelection,
      tokenFileSelection,
    );
    expect(patch.credential_values_to_delete).toEqual(["api_key"]);
    expect(patch.credential_values).toEqual({
      anthropic_federation_rule_id: "fdrl_new",
      anthropic_organization_id: "org-new",
      anthropic_identity_token_file: "/var/run/secrets/anthropic/token",
    });
  });

  it("deletes every stored federation value when the admin switches the credential to an api key", () => {
    const patch = buildCredentialPatch(storedKeycloak, { api_key: "sk-ant-new" }, keycloakSelection, apiKeySelection);
    expect(patch.credential_values).toEqual({ api_key: "sk-ant-new" });
    expect([...patch.credential_values_to_delete].sort()).toEqual(Object.keys(storedKeycloak).sort());
  });

  it("leaves a stored api key alone when the admin keeps a federated credential federated", () => {
    const stored = { ...storedTokenFile, api_key: "sk-a****" };
    const patch = buildCredentialPatch(
      stored,
      { ...storedTokenFile, anthropic_organization_id: "org-edited" },
      tokenFileSelection,
      tokenFileSelection,
    );
    expect(patch.credential_values_to_delete).toEqual([]);
  });

  it("never names one key as both a write and a delete", () => {
    const patch = buildCredentialPatch(
      storedKeycloak,
      { ...storedKeycloak, anthropic_identity_token_file: "/var/run/secrets/anthropic/token" },
      keycloakSelection,
      tokenFileSelection,
    );
    const written = Object.keys(patch.credential_values);
    expect(patch.credential_values_to_delete.filter((key) => written.includes(key))).toEqual([]);
  });
});
