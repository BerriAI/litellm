import type { AgentFormValues, AgentRequestPayload } from "./AgentFormKit";

export interface EntraAgentIdentity {
  provider: "microsoft_entra";
  tenant_id: string;
  client_id: string;
}

export const IDENTITY_UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export const readAgentIdentity = (value: unknown): EntraAgentIdentity | null => {
  if (typeof value !== "object" || value === null) return null;
  if (!("provider" in value) || value.provider !== "microsoft_entra") return null;
  if (!("tenant_id" in value) || typeof value.tenant_id !== "string" || !IDENTITY_UUID_PATTERN.test(value.tenant_id))
    return null;
  if (!("client_id" in value) || typeof value.client_id !== "string" || !IDENTITY_UUID_PATTERN.test(value.client_id))
    return null;
  return { provider: value.provider, tenant_id: value.tenant_id, client_id: value.client_id };
};

export const parseIdentityForForm = (params?: Record<string, unknown> | null): AgentFormValues => {
  const identity = readAgentIdentity(params?.identity);
  return {
    identity_provider: identity?.provider ?? "none",
    identity_tenant_id: identity?.tenant_id ?? "",
    identity_client_id: identity?.client_id ?? "",
  };
};

export const buildIdentityParams = (
  values: AgentFormValues,
  existingIdentity?: unknown,
): { identity?: EntraAgentIdentity | null } => {
  if (values.identity_provider === undefined) return {};
  if (values.identity_provider !== "microsoft_entra")
    return readAgentIdentity(existingIdentity) ? { identity: null } : {};
  const identity = readAgentIdentity({
    provider: "microsoft_entra",
    tenant_id: typeof values.identity_tenant_id === "string" ? values.identity_tenant_id.trim().toLowerCase() : "",
    client_id: typeof values.identity_client_id === "string" ? values.identity_client_id.trim().toLowerCase() : "",
  });
  if (!identity) throw new Error("Enter valid Entra tenant and application client IDs");
  return { identity };
};

export const entraTenantFromIssuer = (issuer: string): string | null => {
  const match = /^https:\/\/login\.microsoftonline\.com\/([^/]+)\/v2\.0$/.exec(issuer);
  return match && IDENTITY_UUID_PATTERN.test(match[1]) ? match[1] : null;
};

export const withAgentIdentity = (
  payload: AgentRequestPayload,
  values: AgentFormValues,
  existingParams?: Record<string, unknown>,
): AgentRequestPayload => {
  const identity = buildIdentityParams(values, existingParams?.identity);
  if (Object.keys(identity).length === 0) return payload;
  return { ...payload, litellm_params: { ...existingParams, ...payload.litellm_params, ...identity } };
};
