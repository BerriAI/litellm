import { AUTH_TYPE, AUTH_TYPE_ITEMS } from "@/components/mcp_tools/types";
import type { AgentCreateInfo } from "@/components/networking";
import type { AgentFormValues, AgentRequestPayload, StaticHeaderFormValue } from "./AgentFormKit";

export const REDACTED_SECRET = "REDACTED_BY_LITELM";

export const AGENT_AUTH_TYPE_ITEMS = AUTH_TYPE_ITEMS.filter(
  (item) => item.value === AUTH_TYPE.NONE || item.value === AUTH_TYPE.BEARER_TOKEN,
);

export interface AgentAuthSupport {
  bearer: boolean;
  headers: boolean;
}

const OWN_AUTH_KEYS = ["custom_llm_provider", "databricks_oauth", "azure_ad_token", "client_secret", "azure_password"];

/** Python truthiness, which the runtime checks: an empty object or list counts as unset. */
const isSet = (value: unknown): boolean =>
  typeof value === "object" && value !== null ? Object.keys(value).length > 0 : Boolean(value);

/** Matches the runtime: an A2A agent with its own provider, Entra or Databricks OAuth credentials never sends the bearer key. */
export const agentAuthSupport = (
  agentType: string,
  usesA2AFormFields?: boolean | null,
  savedParams?: Record<string, unknown> | null,
): AgentAuthSupport => ({
  bearer: agentType === "langgraph" || (agentType === "a2a" && !OWN_AUTH_KEYS.some((key) => isSet(savedParams?.[key]))),
  headers: agentType === "a2a" || agentType === "langgraph" || Boolean(usesA2AFormFields),
});

/** The section owns the bearer key, so the agent type's own api_key field leaves the form. */
export const withoutBearerKeyField = (
  info: AgentCreateInfo | undefined,
  bearer: boolean,
): AgentCreateInfo | undefined =>
  info && bearer ? { ...info, credential_fields: info.credential_fields.filter(({ key }) => key !== "api_key") } : info;

interface SavedAgentAuth {
  litellm_params?: Record<string, unknown> | null;
  static_headers?: Record<string, string> | null;
  extra_headers?: string[] | null;
}

export const agentAuthFormValues = (agent: SavedAgentAuth): AgentFormValues => {
  const apiKey = agent.litellm_params?.api_key;
  const savedKey = typeof apiKey === "string" ? apiKey : "";
  return {
    auth_type: savedKey ? AUTH_TYPE.BEARER_TOKEN : AUTH_TYPE.NONE,
    api_key: savedKey,
    static_headers: Object.entries(agent.static_headers ?? {}).map(([header, value]) => ({ header, value })),
    extra_headers: agent.extra_headers ?? [],
  };
};

/** Converts the form's [{header, value}, ...] rows to {header: value, ...}, skipping rows without a name. */
const staticHeaderMap = (rows: StaticHeaderFormValue[] | undefined): Record<string, string> =>
  Object.fromEntries(
    (rows ?? []).flatMap(({ header, value }) => {
      const name = header?.trim();
      return name ? [[name, value ?? ""]] : [];
    }),
  );

const bearerKey = (values: AgentFormValues): string | undefined =>
  values.auth_type === AUTH_TYPE.BEARER_TOKEN && values.api_key ? values.api_key : undefined;

const sameHeaders = (a: Record<string, string>, b: Record<string, string>): boolean =>
  JSON.stringify(a) === JSON.stringify(b);

/** Adds only the auth fields the admin changed, since PATCH replaces litellm_params as a whole. */
export const withAgentAuth = (
  payload: AgentRequestPayload,
  values: AgentFormValues,
  support: AgentAuthSupport,
  saved: SavedAgentAuth = {},
): AgentRequestPayload => {
  const before = agentAuthFormValues(saved);
  const apiKey = bearerKey(values) ?? null;
  const staticHeaders = staticHeaderMap(values.static_headers);
  const extraHeaders = values.extra_headers ?? [];
  const keyChanged = support.bearer && apiKey !== (bearerKey(before) ?? null);
  const staticChanged = support.headers && !sameHeaders(staticHeaders, staticHeaderMap(before.static_headers));
  const extraChanged = support.headers && JSON.stringify(extraHeaders) !== JSON.stringify(before.extra_headers);
  return {
    ...payload,
    ...(keyChanged ? { litellm_params: { ...(payload.litellm_params ?? saved.litellm_params), api_key: apiKey } } : {}),
    ...(staticChanged ? { static_headers: staticHeaders } : {}),
    ...(extraChanged ? { extra_headers: extraHeaders } : {}),
  };
};

export interface DiscoveryAuth {
  api_key?: string;
  static_headers?: Record<string, string>;
}

export const discoveryAuth = (values: AgentFormValues, support: AgentAuthSupport): DiscoveryAuth => {
  const apiKey = support.bearer ? bearerKey(values) : undefined;
  const staticHeaders = support.headers ? staticHeaderMap(values.static_headers) : {};
  return {
    // A saved key reads back redacted, so only a key typed into the form is sent.
    ...(apiKey?.trim() && apiKey !== REDACTED_SECRET ? { api_key: apiKey } : {}),
    ...(Object.keys(staticHeaders).length > 0 ? { static_headers: staticHeaders } : {}),
  };
};

export const isAuthValueMissing = (value: unknown, values: AgentFormValues): boolean =>
  values.auth_type === AUTH_TYPE.BEARER_TOKEN && !(typeof value === "string" && value.trim());
