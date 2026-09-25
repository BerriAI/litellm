import { describe, expect, it } from "vitest";
import {
  buildIdentityParams,
  entraTenantFromIssuer,
  parseIdentityForForm,
  readAgentIdentity,
  withAgentIdentity,
} from "./agent_identity";

const identity = {
  provider: "microsoft_entra",
  tenant_id: "11111111-1111-4111-8111-111111111111",
  client_id: "22222222-2222-4222-8222-222222222222",
  service_principal_id: "33333333-3333-4333-8333-333333333333",
  required_roles: ["Agent.Invoke"],
  required_scopes: ["user_impersonation"],
} satisfies import("./agent_identity").EntraAgentIdentity;

describe("agent identity configuration", () => {
  it("round trips an existing binding independently of the agent name and runtime", () => {
    const values = {
      ...parseIdentityForForm({
        identity: { ...identity, agent_id: "stable", active: true, revision: "rev", issuer: "https://issuer.example" },
      }),
      agent_name: "Renamed",
      url: "https://new-runtime.example",
    };
    expect(buildIdentityParams(values)).toEqual({ identity });
  });
  it("preserves untouched bindings and explicitly clears a removed binding", () => {
    expect(buildIdentityParams({ agent_name: "legacy" })).toEqual({});
    expect(buildIdentityParams({ identity_provider: "none" }, identity)).toEqual({ identity: null });
    expect(parseIdentityForForm({}).identity_provider).toBe("none");
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
  it("submits identity and budget as top-level settings without changing runtime parameters", () => {
    const formValues = {
      identity_provider: "microsoft_entra",
      identity_tenant_id: identity.tenant_id,
      identity_client_id: identity.client_id,
      identity_service_principal_id: identity.service_principal_id,
      execution_mode: "both",
      enabled: false,
      agent_max_budget: 0,
      agent_budget_duration: "1d",
    };
    const payload = withAgentIdentity({ litellm_params: { model: "runtime" } }, formValues);
    expect(payload.litellm_params).toEqual({ model: "runtime" });
    expect(payload.identity).toMatchObject({
      client_id: identity.client_id,
      service_principal_id: identity.service_principal_id,
    });
    expect(payload.execution_mode).toBe("both");
    expect(payload.enabled).toBe(false);
    expect(payload.budget).toEqual({ max_budget: 0, budget_duration: "1d" });
  });
  it("requires a service principal for autonomous execution", () => {
    const values = {
      identity_provider: "microsoft_entra",
      identity_tenant_id: identity.tenant_id,
      identity_client_id: identity.client_id,
      execution_mode: "autonomous",
    };
    expect(() => buildIdentityParams(values)).toThrow("Enterprise application Object ID");
  });
  it("preserves directory ownership while editing a native agent without an app-only principal", () => {
    const native = { ...identity, service_principal_id: null, provisioning_source_id: "source-one" };
    const values = parseIdentityForForm({
      identity: {
        ...native,
        agent_id: "native-agent",
        active: true,
        revision: "rev",
        issuer: "https://issuer.example",
      },
      execution_mode: "autonomous",
      enabled: false,
    });
    expect(values.identity_provisioning_source_id).toBe("source-one");
    expect(buildIdentityParams(values, native)).toEqual({ identity: native });
  });
  it("only offers tenant-specific Microsoft issuers", () => {
    expect(entraTenantFromIssuer(`https://login.microsoftonline.com/${identity.tenant_id}/v2.0`)).toBe(
      identity.tenant_id,
    );
    expect(entraTenantFromIssuer("https://attacker.example/tenant/v2.0")).toBeNull();
    expect(entraTenantFromIssuer("https://login.microsoftonline.com/common/v2.0")).toBeNull();
  });
});
