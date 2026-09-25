import type { components } from "@/lib/http/schema";
import type { AgentFormValues, AgentRequestPayload } from "./AgentFormKit";

export type EntraAgentIdentity = components["schemas"]["EntraIdentityConfig"];
type AgentIdentityState = Pick<
  components["schemas"]["AgentResponse"],
  "identity" | "enabled" | "execution_mode" | "litellm_budget_table"
>;

export const IDENTITY_UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export const readAgentIdentity = (value: unknown): EntraAgentIdentity | null => {
  if (typeof value !== "object" || value === null) return null;
  if (!("provider" in value) || value.provider !== "microsoft_entra") return null;
  if (!("tenant_id" in value) || typeof value.tenant_id !== "string" || !IDENTITY_UUID_PATTERN.test(value.tenant_id))
    return null;
  if (!("client_id" in value) || typeof value.client_id !== "string" || !IDENTITY_UUID_PATTERN.test(value.client_id))
    return null;
  const principal = "service_principal_id" in value ? value.service_principal_id : null;
  if (principal != null && (typeof principal !== "string" || !IDENTITY_UUID_PATTERN.test(principal))) return null;
  const roles =
    "required_roles" in value && Array.isArray(value.required_roles)
      ? value.required_roles.filter((role): role is string => typeof role === "string")
      : [];
  const scopes =
    "required_scopes" in value && Array.isArray(value.required_scopes)
      ? value.required_scopes.filter((scope): scope is string => typeof scope === "string")
      : ["user_impersonation"];
  return {
    provider: value.provider,
    tenant_id: value.tenant_id,
    client_id: value.client_id,
    service_principal_id: principal,
    required_roles: roles,
    required_scopes: scopes,
  };
};

export const parseIdentityForForm = (agent?: Partial<AgentIdentityState> | null): AgentFormValues => {
  const identity = readAgentIdentity(agent?.identity);
  return {
    identity_provider: identity && agent?.identity?.active !== false ? identity.provider : "none",
    identity_tenant_id: identity?.tenant_id ?? "",
    identity_client_id: identity?.client_id ?? "",
    identity_service_principal_id: identity?.service_principal_id ?? "",
    identity_required_roles: identity?.required_roles?.join(", ") ?? "",
    identity_required_scopes: identity?.required_scopes?.join(", ") ?? "user_impersonation",
    execution_mode: agent?.execution_mode ?? "autonomous",
    enabled: agent?.enabled ?? true,
    agent_max_budget: agent?.litellm_budget_table?.max_budget ?? "",
    agent_budget_duration: agent?.litellm_budget_table?.budget_duration ?? "",
  };
};

const splitGrants = (value: unknown, fallback: string[]): string[] =>
  typeof value === "string"
    ? value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
    : fallback;

export const buildIdentityParams = (
  values: AgentFormValues,
  existingIdentity?: unknown,
): { identity?: EntraAgentIdentity | null } => {
  if (values.identity_provider === undefined) return {};
  if (values.identity_provider !== "microsoft_entra")
    return readAgentIdentity(existingIdentity) ? { identity: null } : {};
  const candidate: EntraAgentIdentity = {
    provider: "microsoft_entra",
    tenant_id: typeof values.identity_tenant_id === "string" ? values.identity_tenant_id.trim().toLowerCase() : "",
    client_id: typeof values.identity_client_id === "string" ? values.identity_client_id.trim().toLowerCase() : "",
    service_principal_id:
      typeof values.identity_service_principal_id === "string" && values.identity_service_principal_id.trim()
        ? values.identity_service_principal_id.trim().toLowerCase()
        : null,
    required_roles: splitGrants(values.identity_required_roles, []),
    required_scopes: splitGrants(values.identity_required_scopes, ["user_impersonation"]),
  };
  const identity = readAgentIdentity(candidate);
  if (!identity) throw new Error("Enter valid Entra tenant, application client and service principal IDs");
  if (values.execution_mode !== "delegated" && !identity.service_principal_id)
    throw new Error("Autonomous agents require the Enterprise application Object ID");
  return { identity };
};

export const entraTenantFromIssuer = (issuer: string): string | null => {
  const match = /^https:\/\/login\.microsoftonline\.com\/([^/]+)\/v2\.0$/.exec(issuer);
  return match && IDENTITY_UUID_PATTERN.test(match[1]) ? match[1] : null;
};

export const withAgentIdentity = (
  payload: AgentRequestPayload,
  values: AgentFormValues,
  existing?: Partial<AgentIdentityState>,
): AgentRequestPayload => {
  const identityFields = buildIdentityParams(values, existing?.identity);
  const managed = values.identity_provider === "microsoft_entra" || Boolean(readAgentIdentity(existing?.identity));
  const budgetIsSet =
    values.agent_max_budget !== undefined && values.agent_max_budget !== "" && values.agent_max_budget !== null;
  const budgetWasSet = existing?.litellm_budget_table?.max_budget != null;
  return {
    ...payload,
    ...identityFields,
    ...(managed && values.execution_mode !== undefined ? { execution_mode: values.execution_mode } : {}),
    ...(managed && values.enabled !== undefined ? { enabled: values.enabled } : {}),
    ...(budgetIsSet
      ? {
          budget: {
            max_budget: Number(values.agent_max_budget),
            budget_duration: values.agent_budget_duration || null,
          },
        }
      : {}),
    ...(!budgetIsSet && budgetWasSet && values.agent_max_budget !== undefined ? { budget: null } : {}),
  };
};
