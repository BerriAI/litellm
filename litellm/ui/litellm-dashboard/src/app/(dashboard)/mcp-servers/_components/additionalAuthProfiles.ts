import type { ParseKeys } from "i18next";
import { v4 as uuidv4 } from "uuid";
import { AUTH_TYPE, MCP_OAUTH2_FLOW_M2M, type AdditionalAuthProfile } from "@/components/mcp_tools/types";
import { ApiError } from "@/lib/http/client";

const DEFAULT_PROFILE_ID = "default";
const PROFILE_ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;

export const PROFILE_KIND = {
  CLIENT_CREDENTIALS: "client_credentials",
  API_KEY: "api_key",
  BEARER_TOKEN: "bearer_token",
  BASIC: "basic",
  TOKEN: "token",
  AUTHORIZATION: "authorization",
} as const;

export type ProfileKind = (typeof PROFILE_KIND)[keyof typeof PROFILE_KIND];

export const PROFILE_KINDS: readonly ProfileKind[] = Object.values(PROFILE_KIND);

export interface AdditionalAuthDraft {
  readonly rowKey: string;
  readonly id: string;
  readonly isNew: boolean;
  readonly kind: ProfileKind;
  readonly client_id: string;
  readonly client_secret: string;
  readonly token_url: string;
  readonly scopes: string;
  readonly token_endpoint_auth_method: string;
  readonly authentication_token: string;
}

export type ProfileProblem = "id_format" | "id_reserved" | "id_duplicate" | "missing_field";

export const newProfileDraft = (): AdditionalAuthDraft => ({
  rowKey: uuidv4(),
  id: "",
  isNew: true,
  kind: PROFILE_KIND.CLIENT_CREDENTIALS,
  client_id: "",
  client_secret: "",
  token_url: "",
  scopes: "",
  token_endpoint_auth_method: "",
  authentication_token: "",
});

export const PROFILE_KIND_LABEL_KEYS: Readonly<Record<ProfileKind, ParseKeys>> = {
  client_credentials: "mcpServers.additionalAuth.kinds.clientCredentials",
  api_key: "mcpTools.authTypes.apiKey",
  bearer_token: "mcpTools.authTypes.bearerToken",
  basic: "mcpTools.authTypes.basic",
  token: "mcpTools.authTypes.token",
  authorization: "mcpTools.authTypes.authorization",
};

export const profileKindOf = (profile: AdditionalAuthProfile): ProfileKind =>
  profile.auth_type === AUTH_TYPE.OAUTH2 ? PROFILE_KIND.CLIENT_CREDENTIALS : (profile.auth_type as ProfileKind);

export const draftFromProfile = (profile: AdditionalAuthProfile): AdditionalAuthDraft => ({
  ...newProfileDraft(),
  id: profile.id,
  isNew: false,
  kind: profileKindOf(profile),
  client_id: profile.client_id ?? "",
  token_url: profile.token_url ?? "",
  scopes: (profile.scopes ?? []).join(" "),
  token_endpoint_auth_method: profile.token_endpoint_auth_method ?? "",
});

export const draftsFromProfiles = (profiles: readonly AdditionalAuthProfile[] | null | undefined) =>
  (profiles ?? []).map(draftFromProfile);

const ifSet = (key: string, value: string): Record<string, string> =>
  value.trim() === "" ? {} : { [key]: value.trim() };

const secretIfSet = (key: string, value: string): Record<string, string> =>
  value.trim() === "" ? {} : { [key]: value };

const splitScopes = (scopes: string): readonly string[] => scopes.split(/\s+/).filter((scope) => scope !== "");

type TokenEndpointAuthMethod = NonNullable<AdditionalAuthProfile["token_endpoint_auth_method"]>;

export const TOKEN_ENDPOINT_AUTH_METHODS: readonly TokenEndpointAuthMethod[] = [
  "client_secret_basic",
  "client_secret_post",
];

const draftToProfile = (draft: AdditionalAuthDraft): AdditionalAuthProfile => {
  if (draft.kind !== PROFILE_KIND.CLIENT_CREDENTIALS) {
    return { id: draft.id, auth_type: draft.kind, ...secretIfSet("authentication_token", draft.authentication_token) };
  }
  const scopes = splitScopes(draft.scopes);
  const tokenEndpointAuthMethod = TOKEN_ENDPOINT_AUTH_METHODS.find(
    (method) => method === draft.token_endpoint_auth_method.trim(),
  );
  return {
    id: draft.id,
    auth_type: "oauth2",
    oauth2_flow: MCP_OAUTH2_FLOW_M2M,
    ...ifSet("client_id", draft.client_id),
    ...secretIfSet("client_secret", draft.client_secret),
    ...ifSet("token_url", draft.token_url),
    ...(scopes.length > 0 || !draft.isNew ? { scopes: [...scopes] } : {}), // an omitted key keeps the stored value
    ...(tokenEndpointAuthMethod || !draft.isNew ? { token_endpoint_auth_method: tokenEndpointAuthMethod ?? null } : {}),
  };
};

export const draftsToProfiles = (drafts: readonly AdditionalAuthDraft[]): readonly AdditionalAuthProfile[] =>
  drafts.map(draftToProfile);

const REQUIRED_WHEN_NEW: Readonly<Record<ProfileKind, readonly (keyof AdditionalAuthDraft)[]>> = {
  client_credentials: ["client_id", "client_secret", "token_url"],
  api_key: ["authentication_token"],
  bearer_token: ["authentication_token"],
  basic: ["authentication_token"],
  token: ["authentication_token"],
  authorization: ["authentication_token"],
};

export const profileProblem = (
  draft: AdditionalAuthDraft,
  drafts: readonly AdditionalAuthDraft[],
): ProfileProblem | null => {
  if (!PROFILE_ID_PATTERN.test(draft.id)) return "id_format";
  if (draft.id.toLowerCase() === DEFAULT_PROFILE_ID) return "id_reserved";
  if (drafts.filter((other) => other.id.toLowerCase() === draft.id.toLowerCase()).length > 1) return "id_duplicate";
  const missing = draft.isNew && REQUIRED_WHEN_NEW[draft.kind].some((field) => String(draft[field]).trim() === "");
  return missing ? "missing_field" : null;
};

export const firstInvalidProfileId = (drafts: readonly AdditionalAuthDraft[]): string | undefined =>
  drafts.find((draft) => profileProblem(draft, drafts) !== null)?.id;

export const additionalAuthEntry = (
  drafts: readonly AdditionalAuthDraft[] | null | undefined,
  { omitWhenEmpty }: { readonly omitWhenEmpty: boolean },
): { readonly additional_auth?: readonly AdditionalAuthProfile[] } => {
  if (drafts === null || drafts === undefined) return {};
  if (omitWhenEmpty && drafts.length === 0) return {};
  return { additional_auth: draftsToProfiles(drafts) };
};

const fieldOf = (value: unknown, key: string): unknown =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>)[key] : undefined;

const holderName = (grant: unknown): string | undefined => {
  const kind = fieldOf(grant, "kind");
  const name = fieldOf(grant, "name");
  return typeof kind === "string" && typeof name === "string" ? `${kind} ${name}` : undefined;
};

export const stillGrantedHolders = (error: unknown): readonly string[] => {
  if (!(error instanceof ApiError) || error.status !== 409) return [];
  const grants = fieldOf(fieldOf(error.body, "detail"), "grants");
  return Array.isArray(grants) ? grants.map(holderName).filter((name): name is string => name !== undefined) : [];
};
