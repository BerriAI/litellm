import { AUTH_TYPE, OAUTH_FLOW, TRANSPORT, isClientForwardedTokenMode } from "@/components/mcp_tools/types";
import { AUTH_TYPES_REQUIRING_AUTH_VALUE } from "./createServerPayload";

export interface MountedFieldNames {
  readonly root: readonly string[];
  readonly credentials: readonly string[];
}

const ENTRA_OBO_PROFILE = "entra_obo";

const ALWAYS_MOUNTED_ROOT = ["server_name", "alias", "description", "transport", "max_concurrent_requests"] as const;

const PERMISSION_SECTION_ROOT = [
  "allow_all_keys",
  "available_on_public_internet",
  "mcp_access_groups",
  "extra_headers",
  "static_headers",
] as const;

const OAUTH_M2M_CREDENTIALS = [
  "client_id",
  "client_secret",
  "token_endpoint_auth_method",
  "scopes",
  "upstream_resource",
] as const;

const OAUTH_INTERACTIVE_CREDENTIALS = [
  "client_id",
  "client_secret",
  "scopes",
  "upstream_resource",
  "token_endpoint_auth_method",
] as const;

const OAUTH_INTERACTIVE_ROOT = [
  "issuer",
  "authorization_url",
  "token_url",
  "registration_url",
  "token_validation_json",
  "token_storage_ttl_seconds",
] as const;

const ID_JAG_CREDENTIALS = [
  "id_jag_resource_token_endpoint",
  "client_id",
  "client_secret",
  "client_private_key",
  "client_private_key_id",
  "client_assertion_signing_alg",
  "id_jag_resource",
  "scopes",
] as const;

const AWS_SIGV4_CREDENTIALS = [
  "aws_region_name",
  "aws_service_name",
  "aws_access_key_id",
  "aws_secret_access_key",
  "aws_session_token",
  "aws_role_name",
  "aws_session_name",
] as const;

const hasAuthorizationExtraHeader = (extraHeaders: unknown): boolean =>
  Array.isArray(extraHeaders) && extraHeaders.some((h) => typeof h === "string" && h.toLowerCase() === "authorization");

interface AuthSubtreeGates {
  readonly authType: string | undefined;
  readonly oauthFlowType: string | undefined;
  readonly tokenExchangeProfile: string | undefined;
}

const authSubtreeRoot = ({ authType, oauthFlowType, tokenExchangeProfile }: AuthSubtreeGates): readonly string[] => {
  if (authType === AUTH_TYPE.OAUTH2) {
    return oauthFlowType === OAUTH_FLOW.M2M
      ? ["oauth_flow_type", "token_url"]
      : ["oauth_flow_type", ...OAUTH_INTERACTIVE_ROOT];
  }
  if (authType === AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE) {
    return tokenExchangeProfile === ENTRA_OBO_PROFILE
      ? ["token_exchange_profile", "token_exchange_endpoint"]
      : ["token_exchange_profile", "token_exchange_endpoint", "audience", "subject_token_type"];
  }
  if (authType === AUTH_TYPE.OAUTH2_ID_JAG) {
    return ["token_exchange_endpoint", "audience", "subject_token_type"];
  }
  return [];
};

const authSubtreeCredentials = ({ authType, oauthFlowType }: AuthSubtreeGates): readonly string[] => {
  const authValue = AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(authType as string) ? ["auth_value"] : [];
  const clientForwarded = isClientForwardedTokenMode(authType) ? ["client_id", "client_secret"] : [];
  if (authType === AUTH_TYPE.OAUTH2) {
    return [
      ...authValue,
      ...(oauthFlowType === OAUTH_FLOW.M2M ? OAUTH_M2M_CREDENTIALS : OAUTH_INTERACTIVE_CREDENTIALS),
    ];
  }
  if (authType === AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE) {
    return [...authValue, "client_id", "client_secret", "scopes"];
  }
  if (authType === AUTH_TYPE.OAUTH2_ID_JAG) {
    return [...authValue, ...ID_JAG_CREDENTIALS];
  }
  if (authType === AUTH_TYPE.AWS_SIGV4) {
    return [...authValue, ...AWS_SIGV4_CREDENTIALS];
  }
  return [...authValue, ...clientForwarded];
};

const permissionSectionRoot = (authType: string | undefined, extraHeaders: unknown): readonly string[] => {
  const isNoneAuth = authType === AUTH_TYPE.NONE || authType == null;
  return [
    ...PERMISSION_SECTION_ROOT,
    ...(authType === AUTH_TYPE.OAUTH2 ? ["delegate_auth_to_upstream"] : []),
    ...(isNoneAuth && hasAuthorizationExtraHeader(extraHeaders) ? ["oauth_passthrough"] : []),
  ];
};

const dedupe = (names: readonly string[]): readonly string[] => Array.from(new Set(names));

export const mountedEditFieldNames = (values: Record<string, unknown>): MountedFieldNames => {
  const transport = values.transport as string | undefined;
  const isStdio = transport === "stdio";
  const isOpenApi = transport === TRANSPORT.OPENAPI;
  const isMcp = !isStdio && !isOpenApi;
  const gates: AuthSubtreeGates = {
    authType: isStdio ? undefined : (values.auth_type as string | undefined),
    oauthFlowType: values.oauth_flow_type as string | undefined,
    tokenExchangeProfile: values.token_exchange_profile as string | undefined,
  };

  return {
    root: dedupe([
      ...ALWAYS_MOUNTED_ROOT,
      ...(isMcp ? ["url"] : []),
      ...(isOpenApi ? ["spec_path"] : []),
      ...(isStdio ? ["command", "args", "env_json", "stdio_config"] : ["auth_type"]),
      ...(isStdio ? [] : authSubtreeRoot(gates)),
      ...(!isStdio && isClientForwardedTokenMode(gates.authType) ? ["dcr_bridge"] : []),
      "env_vars",
      ...permissionSectionRoot(gates.authType, values.extra_headers),
    ]),
    credentials: isStdio ? [] : dedupe(authSubtreeCredentials(gates)),
  };
};

export const mountedCreateFieldNames = (values: Record<string, unknown>): MountedFieldNames => {
  const transport = values.transport as string | undefined;
  const isStdio = transport === "stdio";
  const isOpenApi = transport === TRANSPORT.OPENAPI;
  const authSectionMounted = !isStdio && transport !== "" && transport !== undefined;
  const gates: AuthSubtreeGates = {
    authType: authSectionMounted ? (values.auth_type as string | undefined) : undefined,
    oauthFlowType: values.oauth_flow_type as string | undefined,
    tokenExchangeProfile: values.token_exchange_profile as string | undefined,
  };

  return {
    root: dedupe([
      ...ALWAYS_MOUNTED_ROOT,
      "source_url",
      ...(transport === "http" || transport === "sse" ? ["url"] : []),
      ...(isOpenApi ? ["spec_path", "is_byok"] : []),
      ...(isOpenApi && values.is_byok ? ["byok_description", "byok_api_key_help_url"] : []),
      ...(authSectionMounted ? ["auth_type"] : []),
      ...(authSectionMounted ? authSubtreeRoot(gates) : []),
      ...(authSectionMounted && isClientForwardedTokenMode(gates.authType) ? ["dcr_bridge"] : []),
      ...(isStdio ? ["stdio_config"] : []),
      "env_vars",
      ...permissionSectionRoot(gates.authType, values.extra_headers),
    ]),
    credentials: authSectionMounted ? dedupe(authSubtreeCredentials(gates)) : [],
  };
};

const pickEmitting = (source: Record<string, unknown> | undefined, names: readonly string[]): Record<string, unknown> =>
  Object.fromEntries(names.map((name) => [name, source?.[name]]));

const projectWith =
  (namesOf: (values: Record<string, unknown>) => MountedFieldNames) =>
  (values: Record<string, unknown>): Record<string, unknown> => {
    const names = namesOf(values);
    const credentials = values.credentials as Record<string, unknown> | undefined;
    return {
      ...pickEmitting(values, names.root),
      ...(names.credentials.length > 0 ? { credentials: pickEmitting(credentials, names.credentials) } : {}),
    };
  };

export const projectMountedEditValues = projectWith(mountedEditFieldNames);
export const projectMountedCreateValues = projectWith(mountedCreateFieldNames);
