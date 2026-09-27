import type { components } from "@/lib/http/schema";

export type KillSwitchConfig = components["schemas"]["AgentKillSwitchConfig"];
export type KillSwitchAuth = NonNullable<KillSwitchConfig["auth"]>;
export type KillSwitchMethod = NonNullable<KillSwitchConfig["method"]>;
export type KillSwitchAuthType = KillSwitchAuth["type"] | "none";

export const KILL_SWITCH_METHODS: readonly KillSwitchMethod[] = ["POST", "PUT", "PATCH", "DELETE", "GET"];
export const KILL_SWITCH_AUTH_TYPES: readonly { value: KillSwitchAuthType; label: string }[] = [
  { value: "none", label: "None" },
  { value: "bearer", label: "Bearer token" },
  { value: "api_key", label: "API key header" },
  { value: "basic", label: "Basic auth" },
];

export interface KeyValueFormValue {
  key?: string;
  value?: string;
}

export interface KillSwitchFormValue {
  url?: string;
  method?: KillSwitchMethod;
  headers?: KeyValueFormValue[];
  query_params?: KeyValueFormValue[];
  body?: string;
  auth_type?: KillSwitchAuthType;
  auth_token?: string;
  auth_header_name?: string;
  auth_api_key?: string;
  auth_username?: string;
  auth_password?: string;
}

export const KILL_SWITCH_PANEL_KEY = "kill_switch";

export const EMPTY_KILL_SWITCH_FORM: Readonly<KillSwitchFormValue> = {
  url: "",
  method: "POST",
  headers: [],
  query_params: [],
  body: "",
  auth_type: "none",
};

const pairsToRecord = (pairs: readonly KeyValueFormValue[] | undefined): Record<string, string> =>
  Object.fromEntries(
    (pairs ?? []).map((pair) => [pair.key?.trim() ?? "", pair.value ?? ""] as const).filter(([key]) => key.length > 0),
  );

const recordToPairs = (record: Record<string, string> | undefined | null): KeyValueFormValue[] =>
  Object.entries(record ?? {}).map(([key, value]) => ({ key, value }));

export const parseKillSwitchBody = (text: string | undefined): Record<string, unknown> | null => {
  const trimmed = text?.trim() ?? "";
  if (trimmed.length === 0) return null;
  const parsed: unknown = JSON.parse(trimmed);
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Body must be a JSON object");
  }
  return parsed as Record<string, unknown>;
};

export const validateKillSwitchBody = (text: string | undefined): true | string => {
  try {
    parseKillSwitchBody(text);
    return true;
  } catch (error) {
    return error instanceof Error ? error.message : "Body must be valid JSON";
  }
};

const buildAuth = (form: KillSwitchFormValue): KillSwitchAuth | null => {
  switch (form.auth_type) {
    case "bearer":
      return { type: "bearer", token: form.auth_token ?? "" };
    case "api_key":
      return {
        type: "api_key",
        header_name: form.auth_header_name?.trim() || "X-API-Key",
        api_key: form.auth_api_key ?? "",
      };
    case "basic":
      return { type: "basic", username: form.auth_username ?? "", password: form.auth_password ?? "" };
    default:
      return null;
  }
};

/**
 * `undefined` means the form never touched the kill switch (leave it as is),
 * `null` means the user cleared the URL (remove it), otherwise the config to save.
 */
export const buildKillSwitchFromForm = (form: KillSwitchFormValue | undefined): KillSwitchConfig | null | undefined => {
  if (form === undefined) return undefined;
  const url = form.url?.trim() ?? "";
  if (url.length === 0) return null;
  return {
    url,
    method: form.method ?? "POST",
    headers: pairsToRecord(form.headers),
    query_params: pairsToRecord(form.query_params),
    body: parseKillSwitchBody(form.body),
    auth: buildAuth(form),
  };
};

export const parseKillSwitchForForm = (config: KillSwitchConfig | null | undefined): KillSwitchFormValue => {
  if (!config) return { ...EMPTY_KILL_SWITCH_FORM };
  const auth = config.auth ?? null;
  return {
    url: config.url,
    method: config.method ?? "POST",
    headers: recordToPairs(config.headers),
    query_params: recordToPairs(config.query_params),
    body: config.body ? JSON.stringify(config.body, null, 2) : "",
    auth_type: auth?.type ?? "none",
    auth_token: auth?.type === "bearer" ? auth.token : "",
    auth_header_name: auth?.type === "api_key" ? auth.header_name : "",
    auth_api_key: auth?.type === "api_key" ? auth.api_key : "",
    auth_username: auth?.type === "basic" ? auth.username : "",
    auth_password: auth?.type === "basic" ? auth.password : "",
  };
};
