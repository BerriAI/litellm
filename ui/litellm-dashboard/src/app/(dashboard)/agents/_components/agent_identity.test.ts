import { describe, expect, it } from "vitest";
import { buildIdentityParams, entraTenantFromIssuer, parseIdentityForForm, readAgentIdentity } from "./agent_identity";

const identity = {
  provider: "microsoft_entra",
  tenant_id: "11111111-1111-4111-8111-111111111111",
  client_id: "22222222-2222-4222-8222-222222222222",
} as const;

describe("agent identity configuration", () => {
  it("round trips an existing binding independently of the agent name and runtime", () => {
    const values = { ...parseIdentityForForm({ identity }), agent_name: "Renamed", url: "https://new-runtime.example" };
    expect(buildIdentityParams(values)).toEqual({ identity });
  });
  it("preserves untouched bindings and explicitly clears a removed binding", () => {
    expect(buildIdentityParams({ agent_name: "legacy" })).toEqual({});
    expect(buildIdentityParams({ identity_provider: "none" }, identity)).toEqual({ identity: null });
    expect(parseIdentityForForm({ model: "model" }).identity_provider).toBe("none");
  });
  it.each([
    null,
    {},
    "invalid",
    { ...identity, client_id: "bad" },
    { ...identity, tenant_id: 3 },
    { ...identity, provider: "other" },
  ])("rejects malformed bindings: %j", (value) => {
    expect(readAgentIdentity(value)).toBeNull();
  });
  it("rejects incomplete submissions", () => {
    expect(() => buildIdentityParams({ identity_provider: "microsoft_entra" })).toThrow("Enter valid Entra");
  });
  it("only offers tenant-specific Microsoft issuers", () => {
    expect(entraTenantFromIssuer(`https://login.microsoftonline.com/${identity.tenant_id}/v2.0`)).toBe(
      identity.tenant_id,
    );
    expect(entraTenantFromIssuer("https://attacker.example/tenant/v2.0")).toBeNull();
    expect(entraTenantFromIssuer("https://login.microsoftonline.com/common/v2.0")).toBeNull();
  });
});
