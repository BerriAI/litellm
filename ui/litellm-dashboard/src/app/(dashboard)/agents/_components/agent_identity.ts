import { z } from "zod";
import type { components } from "@/lib/http/schema";
import type { AgentFormValues, AgentRequestPayload } from "./AgentFormKit";

export type EntraAgentIdentity = components["schemas"]["EntraIdentityConfig"];
export type OktaAgentIdentity = components["schemas"]["OktaIdentityConfig"];
export type AgentIdentity = EntraAgentIdentity | OktaAgentIdentity;
type AgentIdentityState = Pick<
  components["schemas"]["AgentResponse"],
  "identity" | "enabled" | "execution_mode" | "agent_card_params"
>;

export const IDENTITY_UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const isBoundIdentityProvider = (provider: unknown): boolean =>
  provider === "microsoft_entra" || provider === "okta";
export const OKTA_ISSUER_PATTERN = /^https:\/\/[^/?#]+\/oauth2\/[^/?#]+$/;

const stringGrants = (fallback: string[]) =>
  z
    .unknown()
    .optional()
    .transform((value) =>
      Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : fallback,
    );

const identityShape = {
  provider: z.literal("microsoft_entra"),
  tenant_id: z.string().regex(IDENTITY_UUID_PATTERN),
  client_id: z.string().regex(IDENTITY_UUID_PATTERN),
  service_principal_id: z.string().regex(IDENTITY_UUID_PATTERN).nullable().default(null),
  required_roles: stringGrants([]),
  required_scopes: stringGrants(["user_impersonation"]),
};
const oktaShape = {
  provider: z.literal("okta"),
  issuer: z.string().regex(OKTA_ISSUER_PATTERN),
  client_id: z.string().min(1),
  required_scopes: stringGrants([]),
};
const identitySchema = z.discriminatedUnion("provider", [z.object(identityShape), z.object(oktaShape)]);

export const readAgentIdentity = (value: unknown): AgentIdentity | null => {
  const parsed = identitySchema.safeParse(value);
  return parsed.success ? parsed.data : null;
};

const oktaFormFields = (identity: OktaAgentIdentity): AgentFormValues => ({
  identity_provider: "okta",
  identity_okta_issuer: identity.issuer,
  identity_okta_client_id: identity.client_id,
  identity_okta_required_scopes: identity.required_scopes?.join(", ") ?? "",
});

const entraFormFields = (identity: EntraAgentIdentity | null): AgentFormValues => ({
  identity_provider: identity?.provider ?? "none",
  identity_tenant_id: identity?.tenant_id ?? "",
  identity_client_id: identity?.client_id ?? "",
  identity_service_principal_id: identity?.service_principal_id ?? "",
  identity_required_roles: identity?.required_roles?.join(", ") ?? "",
  identity_required_scopes: identity?.required_scopes?.join(", ") ?? "user_impersonation",
});

const identityFormFields = (identity: AgentIdentity | null): AgentFormValues =>
  identity?.provider === "okta" ? oktaFormFields(identity) : entraFormFields(identity);

export const parseIdentityForForm = (agent?: Partial<AgentIdentityState> | null): AgentFormValues => {
  const identity = agent?.identity?.active === false ? null : readAgentIdentity(agent?.identity);
  return {
    ...identityFormFields(identity),
    execution_mode: agent?.execution_mode ?? "autonomous",
    enabled: agent?.enabled ?? true,
  };
};

const splitGrants = (value: unknown, fallback: string[]): string[] =>
  typeof value === "string"
    ? value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
    : fallback;

const splitScopes = (value: unknown): string[] =>
  typeof value === "string" ? value.split(/[\s,]+/).filter(Boolean) : [];

const trimmed = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

const buildOktaIdentity = (values: AgentFormValues): OktaAgentIdentity | null => {
  const candidate: OktaAgentIdentity = {
    provider: "okta",
    issuer: trimmed(values.identity_okta_issuer),
    client_id: trimmed(values.identity_okta_client_id),
    required_scopes: splitScopes(values.identity_okta_required_scopes),
  };
  const identity = readAgentIdentity(candidate);
  return identity?.provider === "okta" ? identity : null;
};

export const buildIdentityParams = (
  values: AgentFormValues,
  existingIdentity?: unknown,
): { identity?: AgentIdentity | null } => {
  if (values.identity_provider === undefined) return {};
  if (values.identity_provider === "okta") {
    const identity = buildOktaIdentity(values);
    if (!identity) throw new Error("Select a trusted Okta issuer and enter the application client ID");
    return { identity };
  }
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
  if (identity?.provider !== "microsoft_entra")
    throw new Error("Enter valid Entra tenant, application client and service principal IDs");
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
  cardEdited = false,
): AgentRequestPayload => {
  const { agent_card_params, ...settings } = payload;
  const hasCard = !existing || cardEdited || Object.keys(existing.agent_card_params ?? {}).length > 0;
  const identityFields = buildIdentityParams(values, existing?.identity);
  const managed = isBoundIdentityProvider(values.identity_provider) || Boolean(readAgentIdentity(existing?.identity));
  const mode = values.identity_provider === "okta" ? "autonomous" : values.execution_mode;
  return {
    ...settings,
    ...(hasCard && agent_card_params ? { agent_card_params } : {}),
    ...identityFields,
    ...(managed && mode !== undefined ? { execution_mode: mode } : {}),
    ...(managed && values.enabled !== undefined ? { enabled: values.enabled } : {}),
  };
};
